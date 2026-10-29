"""Checks for pattern 08: Observer, callbacks, hooks and pub/sub."""
import logging
import sqlite3
import threading
import time
import types

import numpy as np
from harness import bug, fine

L1 = "08_observer/L1_event_bus"
L2 = "08_observer/L2_training_callbacks"
L3 = "08_observer/L3_job_events"


# --- L1: event bus -------------------------------------------------------------------


@bug(L1)
def test_L1_late_binding_counter(m):
    """Blocker 1: the handler closes over the loop variable, so every event counts as 'failed'."""
    bus = m.EventBus()
    counter = m.JobStateCounter(bus)
    bus.publish("job.queued", {"job_id": "j1"})
    bus.publish("job.running", {"job_id": "j1"})
    assert counter.counts == {"queued": 1, "running": 1}


@bug(L1)
def test_L1_subscribers_shared_between_buses(m):
    """Blocker 2: _subscribers is a class attribute, so every bus delivers every other bus's events."""
    bus_a, bus_b = m.EventBus(), m.EventBus()
    store_a: dict[str, float] = {}
    m.ResultsWriter(bus_a, store_a)
    bus_b.publish("job.finished", {"job_id": "other-campaign", "uptake": 9.9})
    assert store_a == {}


@bug(L1)
def test_L1_once_skips_next_handler(m):
    """Blocker 3: once() unsubscribes during publish, so the next handler misses the event."""
    bus = m.EventBus()
    notified: list[str] = []
    bus.once("job.finished", lambda e: notified.append(e["job_id"]))
    store: dict[str, float] = {}
    m.ResultsWriter(bus, store)
    bus.publish("job.finished", {"job_id": "j1", "uptake": 2.1})
    assert notified == ["j1"]
    assert store == {"j1": 2.1}


@bug(L1, raises=RuntimeError)
def test_L1_handler_error_stops_delivery(m):
    """Major 4: one handler's exception skips the remaining handlers and reaches the publisher."""
    bus = m.EventBus()

    def broken(event):
        raise RuntimeError("dashboard is down")

    bus.subscribe("job.finished", broken)
    store: dict[str, float] = {}
    m.ResultsWriter(bus, store)
    bus.publish("job.finished", {"job_id": "j1", "uptake": 2.1})
    assert store == {"j1": 2.1}


@bug(L1)
def test_L1_publish_grows_topic_table(m):
    """Minor 5: publishing to a topic nobody listens to adds an entry that is never removed."""
    bus = m.EventBus()
    for i in range(100):
        bus.publish(f"job.{i}.heartbeat", {"job_id": str(i)})
    assert len(bus._subscribers) == 0


@fine(L1)
def test_L1_unsubscribe_bound_method(m):
    """Red herring: unsubscribing self.on_finished works, because bound methods compare equal."""
    bus = m.EventBus()
    store: dict[str, float] = {}
    writer = m.ResultsWriter(bus, store)
    assert writer.on_finished == writer.on_finished and writer.on_finished is not writer.on_finished
    writer.close()
    bus.publish("job.finished", {"job_id": "j1", "uptake": 2.1})
    assert store == {}


# --- L2: training callbacks ------------------------------------------------------------


def _data(n=64, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    y = X @ np.array([1.0, -2.0, 0.5]) + 0.1 * rng.normal(size=n)
    return X, y


def _fake_trainer(weights=(0.0, 0.0, 0.0)):
    return types.SimpleNamespace(stop_training=False, model=types.SimpleNamespace(weights=np.array(weights)))


@bug(L2)
def test_L2_history_aliases_logs(m):
    """Blocker 1: one logs dict is reused, so every History entry shows the last epoch."""
    X, y = _data()
    history = m.History()
    m.Trainer(m.LinearSurrogate(3), [history]).fit(X, y, X, y, epochs=3)
    val = [entry["val_loss"] for entry in history.epochs]
    assert val[0] > val[-1]


@bug(L2)
def test_L2_stop_flag_not_reset(m):
    """Blocker 2: stop_training is never reset, so the next fit runs a single epoch."""
    X, y = _data()

    class StopNow(m.Callback):
        def on_epoch_end(self, trainer, epoch, logs):
            trainer.stop_training = True

    history = m.History()
    trainer = m.Trainer(m.LinearSurrogate(3), [StopNow(), history])
    trainer.fit(X, y, X, y, epochs=5)
    del trainer.callbacks[0]  # the next round trains without the stopper
    trainer.fit(X, y, X, y, epochs=5)
    assert len(history.epochs) == 1 + 5


@bug(L2)
def test_L2_checkpoint_uses_validation(m, tmp_path):
    """Blocker 3: the checkpoint keeps the lowest batch training loss, not the best val_loss."""
    path = str(tmp_path / "best.npy")
    ckpt = m.ModelCheckpoint(path)
    trainer = _fake_trainer()
    ckpt.on_train_begin(trainer)
    trainer.model.weights = np.array([1.0, 1.0, 1.0])  # epoch 0 generalises well
    ckpt.on_batch_end(trainer, 0, {"loss": 0.50})
    ckpt.on_epoch_end(trainer, 0, {"loss": 0.50, "val_loss": 0.20})
    trainer.model.weights = np.array([9.0, 9.0, 9.0])  # epoch 1 fits one easy batch
    ckpt.on_batch_end(trainer, 0, {"loss": 0.01})
    ckpt.on_epoch_end(trainer, 1, {"loss": 0.01, "val_loss": 0.90})
    assert np.array_equal(np.load(path), [1.0, 1.0, 1.0])


@bug(L2)
def test_L2_early_stopping_mode_max(m):
    """Major 4: mode='max' is ignored, so a rising R² counts as no improvement."""
    es = m.EarlyStopping(monitor="val_r2", patience=2, mode="max")
    trainer = _fake_trainer()
    es.on_train_begin(trainer)
    for epoch, r2 in enumerate([0.1, 0.2, 0.3, 0.4]):
        es.on_epoch_end(trainer, epoch, {"val_r2": r2})
    assert trainer.stop_training is False


@bug(L2)
def test_L2_callback_state_not_reset(m):
    """Major 5: EarlyStopping keeps round 1's best, so round 2 stops while still improving."""
    es = m.EarlyStopping(patience=2)
    trainer = _fake_trainer()
    es.on_train_begin(trainer)
    for epoch, loss in enumerate([0.10, 0.05]):  # round 1
        es.on_epoch_end(trainer, epoch, {"val_loss": loss})
    es.on_train_begin(trainer)  # round 2: new labels, new validation set
    for epoch, loss in enumerate([0.30, 0.25, 0.20]):
        es.on_epoch_end(trainer, epoch, {"val_loss": loss})
    assert trainer.stop_training is False


@bug(L2)
def test_L2_checkpoint_writes_every_batch(m, tmp_path, monkeypatch):
    """Major 6: the checkpoint writes to disk inside the batch loop."""
    saves = []
    real_save = np.save
    monkeypatch.setattr(m.np, "save", lambda *a, **k: (saves.append(1), real_save(*a, **k)))
    X, y = _data(n=256)
    trainer = m.Trainer(m.LinearSurrogate(3), [m.ModelCheckpoint(str(tmp_path / "best.npy"))])
    trainer.fit(X, y, X, y, epochs=2, batch_size=8)
    assert len(saves) <= 2


@bug(L2)
def test_L2_stop_checked_after_epoch(m):
    """Minor 7: a stop requested in on_batch_end is ignored until the epoch ends."""
    X, y = _data()

    class StopAtFirstBatch(m.Callback):
        batches = 0

        def on_batch_end(self, trainer, batch, logs):
            self.batches += 1
            trainer.stop_training = True

    cb = StopAtFirstBatch()
    m.Trainer(m.LinearSurrogate(3), [cb]).fit(X, y, X, y, epochs=3, batch_size=8)
    assert cb.batches == 1


@bug(L2)
def test_L2_checkpoint_path_suffix(m, tmp_path):
    """Minor 8: np.save appends .npy, so the checkpoint isn't where the caller reloads it."""
    path = tmp_path / "best_weights"
    X, y = _data()
    m.Trainer(m.LinearSurrogate(3), [m.ModelCheckpoint(str(path))]).fit(X, y, X, y, epochs=2)
    assert path.exists()


@fine(L2)
def test_L2_callbacks_default_not_shared(m):
    """Red herring: `callbacks or []` gives each Trainer its own list."""
    t1, t2 = m.Trainer(m.LinearSurrogate(3)), m.Trainer(m.LinearSurrogate(3))
    t1.callbacks.append(m.History())
    assert t2.callbacks == []


# --- L3: job-status pub/sub ----------------------------------------------------------


class _Recorder:
    """A subscriber that records (job_id, status, payload) and can signal a final status."""

    def __init__(self, until: str | None = None):
        self.seen: list[tuple[str, str, dict]] = []
        self.done = threading.Event()
        self.until = until

    def __call__(self, event):
        self.seen.append((event.job_id, event.status, dict(event.payload)))
        if event.status == self.until:
            self.done.set()


def _rows(path):
    with sqlite3.connect(path) as conn:
        return conn.execute("SELECT job_id, uptake, raw_output FROM results").fetchall()


@bug(L3)
def test_L3_bound_method_subscriber_dropped(m):
    """Blocker 1: weakref.ref to a bound method dies at once, so RetrainTrigger never hears anything."""
    events = m.JobEvents()
    trigger = m.RetrainTrigger(events, retrain=lambda: None, every=2)
    events.emit(m.JobEvent("j1", "finished", {"uptake": 2.1}))
    assert trigger.n_new == 1


@bug(L3)
def test_L3_payload_mutated_by_subscriber(m, tmp_path):
    """Blocker 2: ProgressLog pops raw_output from the shared payload before ResultsStore saves it."""
    events = m.JobEvents()
    log = m.ProgressLog(events)
    store = m.ResultsStore(events, str(tmp_path / "results.db"))  # noqa: F841 (the hub holds only a weak ref)
    events.emit(m.JobEvent("j1", "finished", {"uptake": 2.1, "raw_output": "GCMC log ..."}))
    assert _rows(tmp_path / "results.db") == [("j1", 2.1, "GCMC log ...")]
    assert len(log.entries) == 1


@bug(L3, raises=RuntimeError)
def test_L3_subscribe_during_emit(m):
    """Blocker 3: emit iterates the live dict, so a subscribe from another thread breaks it."""
    events = m.JobEvents()

    def late_subscriber(event):
        pass

    def slow_dashboard(event):  # while it runs, another thread subscribes to a new pattern
        t = threading.Thread(target=events.subscribe, args=("fail*", late_subscriber))
        t.start()
        t.join()

    events.subscribe("finished", slow_dashboard)
    events.emit(m.JobEvent("j1", "finished", {"uptake": 2.1}))


@bug(L3)
def test_L3_sqlite_used_from_worker_thread(m, tmp_path):
    """Blocker 4: ResultsStore's connection is used from worker threads, so every insert fails."""
    events = m.JobEvents()
    store = m.ResultsStore(events, str(tmp_path / "results.db"))  # noqa: F841 (the hub holds only a weak ref)
    runner = m.JobRunner(lambda job_id, params: {"uptake": 2.1, "raw_output": "ok"}, events, n_workers=2)
    runner.submit("j1", {"cif": "ZIF-8.cif"})
    runner.shutdown()
    assert _rows(tmp_path / "results.db") == [("j1", 2.1, "ok")]


@bug(L3)
def test_L3_subscriber_errors_swallowed(m, caplog):
    """Major 5: subscriber exceptions are swallowed with no log."""
    events = m.JobEvents()

    def broken(event):
        raise sqlite3.OperationalError("database is locked")

    events.subscribe("finished", broken)
    with caplog.at_level(logging.ERROR):
        events.emit(m.JobEvent("j1", "finished", {"uptake": 2.1}))
    assert "database is locked" in caplog.text


@bug(L3)
def test_L3_retrain_runs_on_worker_thread(m):
    """Major 6: RetrainTrigger retrains synchronously on the thread that emitted the event."""
    ran_on, done = [], threading.Event()

    def retrain():
        ran_on.append(threading.get_ident())
        done.set()

    trigger = m.RetrainTrigger(m.JobEvents(), retrain, every=1)
    trigger.on_finished(m.JobEvent("j1", "finished", {"uptake": 2.1}))
    assert done.wait(timeout=2)
    assert ran_on[0] != threading.get_ident()


@bug(L3)
def test_L3_reentrant_emit_order(m):
    """Major 7: a retry emitted from inside a callback is delivered before the 'failed' event."""
    events = m.JobEvents()
    calls = []

    def flaky(job_id, params):
        calls.append(job_id)
        if len(calls) == 1:
            raise RuntimeError("SCF did not converge")
        return {"uptake": 2.1}

    runner = m.JobRunner(flaky, events, n_workers=1)
    retry = m.RetryOnFailure(runner, max_retries=1)  # subscribed before the recorder
    recorder = _Recorder(until="finished")
    events.subscribe("*", recorder)
    runner.submit("j1", {"cif": "ZIF-8.cif"})
    assert recorder.done.wait(timeout=2)
    runner.shutdown()
    statuses = [status for _, status, _ in recorder.seen]
    assert statuses == ["queued", "running", "failed", "queued", "running", "finished"]
    assert retry.attempts["j1"] == 1


@bug(L3)
def test_L3_dead_refs_never_pruned(m):
    """Minor 9: dead weak references stay in the subscriber lists forever."""
    events = m.JobEvents()

    class Listener:
        def __call__(self, event):
            pass

    for _ in range(100):
        events.subscribe("finished", Listener())  # nothing else references it, so it dies
    events.emit(m.JobEvent("j1", "finished", {"uptake": 2.1}))
    assert sum(len(refs) for refs in events._subscribers.values()) == 0


@bug(L3)
def test_L3_runner_crash_vanishes(m, caplog):
    """Minor 10: an exception that escapes _run is stored in a future nobody reads."""
    events = m.JobEvents()
    recorder = _Recorder()
    events.subscribe("failed", recorder)

    def cli_style_wrapper(job_id, params):
        raise SystemExit("gcmc_sim: missing --structure")

    runner = m.JobRunner(cli_style_wrapper, events, n_workers=1)
    with caplog.at_level(logging.ERROR):
        runner.submit("j1", {})
        runner.shutdown()
    assert "j1" in caplog.text
    assert [status for _, status, _ in recorder.seen] == ["failed"]


@bug(L3)
def test_L3_failure_detail_lost(m):
    """Minor 11: str(exc) turns KeyError('cif') into "'cif'", losing the exception type."""
    events = m.JobEvents()
    recorder = _Recorder(until="failed")
    events.subscribe("failed", recorder)

    def run_fn(job_id, params):
        return {"uptake": params["cif"]}

    runner = m.JobRunner(run_fn, events, n_workers=1)
    runner.submit("j1", {})
    runner.shutdown()
    assert "KeyError" in recorder.seen[0][2]["error"]


@fine(L3)
def test_L3_unsubscribe_uses_equality(m):
    """Red herring: unsubscribe matches with ==, which also works for bound methods."""
    events = m.JobEvents()
    trigger = m.RetrainTrigger(events, retrain=lambda: None, every=10)
    assert trigger.on_finished == trigger.on_finished and trigger.on_finished is not trigger.on_finished
    recorder = _Recorder()
    events.subscribe("finished", recorder)
    events.unsubscribe("finished", recorder)
    events.emit(m.JobEvent("j1", "finished", {"uptake": 2.1}))
    assert recorder.seen == []


@fine(L3)
def test_L3_timestamp_default_factory(m):
    """Red herring: field(default_factory=time.time) stamps each event when it's created."""
    first = m.JobEvent("j1", "queued")
    time.sleep(0.01)
    second = m.JobEvent("j1", "running")
    assert second.timestamp > first.timestamp


@fine(L3)
def test_L3_progress_log_bounded(m):
    """Red herring: deque(maxlen=...) keeps ProgressLog bounded, and append is thread-safe."""
    events = m.JobEvents()
    log = m.ProgressLog(events, maxlen=3)
    for i in range(5):
        events.emit(m.JobEvent(f"j{i}", "queued", {"cif": f"{i}.cif"}))
    assert [e.job_id for e in log.entries] == ["j2", "j3", "j4"]
