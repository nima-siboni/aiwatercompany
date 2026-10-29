# Answer key: 18_prediction_service.py

**Good clarifying questions:** Exactly what feature pipeline did training use (feature order,
log base, pressure unit)? Does `UptakeMLP` contain BatchNorm? How many workers and replicas,
and how do clients poll `/screen`? What request sizes and rates do we expect from the dashboard
and from the agents, which may send odd inputs? What happens to the prediction log afterwards?

## Blockers

1. **The service builds different features from the ones the model was trained on, and
   returns log units** (`featurise`, `predict_queries`). Three separate problems:
   - **Pressure.** The model was trained on log10(P / Pa); the service feeds ln(P / bar). At
     0.15 bar that is −1.90 instead of 4.18, far outside the training range.
   - **Descriptor order.** `list(q.descriptors.values())` follows the client's JSON key order,
     while the scaler and the model expect the training order. The same MOF with its keys in
     another order gave 1.151 instead of 0.297 (log10 units). Nothing checks for missing or
     extra keys either.
   - **No back-transform.** The model predicts log10(uptake), and the service returns that
     number as `uptake_mmol_g`: 0.297 instead of 1.98 mmol/g. The std of log10 samples isn't a
     std in mmol/g either.
   *Fix:* one `featurise` shipped and versioned with the checkpoint, used by both training and
   serving; index the descriptors by the training feature list and reject missing or unknown
   keys; transform each MC sample back before taking the mean and std. Add a parity test: a
   training row sent through the HTTP layer must reproduce the training prediction.

2. **MC dropout via `model.train()` on a shared model with BatchNorm** (`mc_dropout`).
   `train()` switches dropout on, but it also makes BatchNorm use batch statistics and update
   its running statistics. That has four consequences:
   - **The deployed model changes with traffic.** After one request the BN running mean had
     moved by 3.1, and the eval-mode prediction for a fixed input went from 0.297 to 0.195, then
     to 0.169 after 20 more requests.
   - **Each prediction depends on the other rows in the request.** One NaN row turned every
     prediction in the request into `null`.
   - **A one-row request returns 500**, with "Expected more than 1 value per channel when
     training".
   - **The model is shared by the thread pool**, so one request's `train()` and `eval()` flip
     the mode under another request.
   *Fix:* keep the model in `eval()` and apply dropout explicitly in a dedicated MC forward
   (functional dropout with `training=True`). Mode flags are shared state, so don't toggle them
   per request. Or use an ensemble.

3. **Deduplicated predictions zipped with the original list** (`predict_queries`). The
   predictions are for the unique queries, but they are zipped with all the queries. Four
   queries [A, B, A, C] gave three predictions, labelled A, B, A: the third carries C's number,
   and C is missing. And because deduplication can leave a single row, the same MOF sent twice
   returns 500 (see 2). *Fix:* map the results back by key, e.g.
   `[by_key[(q.mof_id, q.pressure_bar, q.temperature_k)] for q in queries]`.

4. **One SQLAlchemy `Session` shared by every request.** FastAPI runs `def` endpoints in a
   thread pool, and a Session isn't thread-safe. Of 200 concurrent requests, all 200 failed
   (PendingRollbackError, IllegalStateChangeError and others), and 0 of the expected 400 log
   rows were stored. The next three requests failed too: the session stays broken until the
   process restarts. The first burst of load takes the service down and loses the monitoring
   data. *Fix:* one session per request, from a `yield` dependency (`with Session(engine) as
   s:`), with the engine shared.

## Major

5. **Request validation is laxer than it looks.**
   - Unknown keys are ignored, so a misspelt `temprature_k` gives 200, computed at the 298 K
     default.
   - `pressure_bar=-1` becomes NaN and returns 200 with `null`, for every row in the request.
   - `pressure_bar=0` returns 500, because the scaler rejects `inf`.
   - A bare `NaN` in the JSON is accepted and comes back as `null`.
   - The list has no size limit: 30 MC passes over 100k rows.
   FastAPI ≥0.130 serialises NaN as `null` with status 200, so none of this is visible to the
   client. *Fix:* `ConfigDict(extra="forbid")`, `Field(gt=0, allow_inf_nan=False)`, bounds on
   the temperature, a typed descriptor model with the training feature names, `max_length` on
   the list, and an explicit error for non-finite predictions.
6. **Job state per worker process.** With `--workers 4`, `JOBS` lives in one worker's memory.
   `GET /screen/{id}` reaches a random worker, so it returns 404 about three times in four, and
   a restart loses every job. *Fix:* keep job state in the database or a queue.
7. **`/health` waits in the same thread pool.** It's a `def`, so under load it queues for one
   of the 40 threads. With 60 slow predictions in flight it answered after 1.35 s, and the
   default Kubernetes probe timeout is 1 s, so pods get restarted exactly when they are busy.
   *Fix:* `async def health()`, with separate readiness and liveness probes.

## Minor

8. **`torch.set_grad_enabled(False)` in the lifespan is thread-local.** Inside the `def`
   endpoints grad mode was on. That's harmless here because the MC passes run under `no_grad`,
   but the line is misleading. Use `torch.inference_mode()` in the inference function.
9. **The model and the scaler are separate artefacts**, matched only by a version string. Load
   them as one bundle that also stores the feature list, and check it.
10. **The 30 MC passes run one after another.** Repeat the batch 30 times in one forward pass.
11. **`run_screen` keeps every prediction in memory**, and uses the same shared session.
12. **The engine is created at import time,** so importing the module needs Postgres. The log
    also doesn't record the request's inputs, so a bad prediction can't be debugged.

## Red herrings (say why they're fine)

- **Plain `def` endpoints doing blocking inference** are right: FastAPI runs them in a thread
  pool. The problem is the shared mutable state (the session, the model's mode), not the `def`.
- **`torch.load(..., weights_only=True)`** is correct and safe. **`joblib.load`** of our own
  scaler is fine too; the pickle risk is for untrusted files.
- **`response_model=list[Prediction]` with floats from `.tolist()`** is fine.
- **`HTTPException(404)` for an unknown job** is the right response. The bug is where `JOBS`
  lives.
- **The 298 K default** is reasonable as a default. The problem is that a misspelt key falls
  back to it silently (5).

## Model 1-minute summary
"Every prediction is wrong before concurrency even comes into it. The pressure feature is the
natural log of bar where training used log10 of pascals, the descriptor order follows the
client's JSON, and the log10 output is returned as mmol/g. MC dropout via `model.train()` also
puts BatchNorm into batch statistics, so the shared model drifts with every request, each
prediction depends on the rest of the batch, and a one-row request crashes. The deduplication
zips the results with the original list, so numbers land on the wrong MOFs. Under load, the
shared Session fails every request and stays broken until a restart. After those: strict
validation, job state in the database, and an async health check. I'd ask for a parity test
through the HTTP layer, a request with duplicates and reordered keys, and a concurrency test."
