# Answer key: 15_crystal_diffusion.py

**Good clarifying questions:** Which guidance convention do we use (is w = 0 unconditional, or
conditional)? What share of the training set is metallic, with a band gap of exactly 0? Does
the denoiser have dropout or other train-only behaviour? How many samples per call, and how
long may sampling take? Has the sampler been tested on data with a known distribution?

## Blockers

1. **One key for three random draws** (`loss_fn`). `t`, `eps_f` and `eps_l` all come from the
   same `key`. On JAX 0.12, draws of different shapes from one key share a prefix: the lattice
   noise was exactly the first 72 values of the coordinate noise. The network sees coordinate
   noise that determines the lattice noise, a shortcut that doesn't exist at sampling time.
   *Fix:* `k_t, k_f, k_l = jax.random.split(key, 3)`.

2. **The condition dropout is decided once, at trace time** (`train_step`). `np.random.rand()`
   runs in Python while JAX traces the function, and the compiled step reuses the result
   forever. Over 300 jitted steps the condition was never dropped, so the unconditional
   branch, which guidance needs, never trains. With probability 0.1 the opposite happens and
   the condition is dropped every step. Even outside `jit`, the drop applies to a whole batch
   rather than to each sample. *Fix:* draw inside the function from a key:
   `drop = jax.random.bernoulli(k_drop, P_UNCOND, (B,))`.

3. **The guidance halves are swapped** (`guided`). The batch is built as [null, gap], so the
   first half of the output is the unconditional one, but the code unpacks it as (conditional,
   unconditional). Guidance then pushes *away* from the requested band gap. In a Gaussian test
   with target mean 2.0 and w = 2, the PR's samples averaged −2.14 instead of 4.00 (guided eps
   0.922, against a correct −0.068). *Fix:* unpack in the order the batch was built, or better,
   return named outputs.

4. **The sampler uses the same key at every step** (`sample`). `jax.random.normal(key, ...)`
   gets the same `key` at all 1,000 steps, for both the lattice and the coordinates. The same
   noise is added again and again, so it accumulates instead of averaging out: in the Gaussian
   test (target std 3.0) the samples had std 81.5. The lattice and coordinate noises are copies
   of each other here too. *Fix:* `key, k_l, k_f = jax.random.split(key, 3)` at every step, or
   carry the key through `lax.scan`.

5. **The posterior variance is used as a standard deviation.** `mean + POST_VAR[t] * z` should
   be `mean + jnp.sqrt(POST_VAR[t]) * z`. On its own, this made samples ten times too tight:
   std 0.298 against 3.0.

## Major

6. **The null condition is a real value.** `NULL_GAP = 0.0` eV is the band gap of every metal,
   so "unconditional" and "metal" are the same input. The model can't learn an unconditional
   distribution separate from metals, and guiding towards metals means nothing. *Fix:* a
   separate learned null embedding, or a mask flag, as MatterGen does.
7. **The coordinate target ignores the wrapping.** −ε/σ is the score of a Gaussian on the real
   line. On the torus the noise is a wrapped normal, whose score differs once σ is a sizeable
   fraction of the cell. The relative RMS error of −ε/σ was 0% at σ = 0.1, 195% at σ = 0.3 and
   3,130% at σ = 0.5, and `SIGMAS` goes up to 0.5. *Fix:* the wrapped-normal score summed over
   periodic images, as in DiffCSP.
8. **The "EMA" is just the current weights.** `optax.incremental_update(new, old, step_size)`
   returns `step_size * new + (1 - step_size) * old`, so `step_size=0.999` keeps 99.9% of the
   new weights (0.999 in the check). *Fix:* `step_size=1 - EMA_DECAY`.
9. **Sampling runs with `train=True`.** Dropout and any other train-only behaviour stay on, so
   the denoiser is noisy and doesn't match the trained model. *Fix:* `train=False` in
   `guided`.
10. **The timestep is a static argument.** Every value of t compiles a new version of `guided`:
    1,001 versions after one sampling run. With a toy denoiser, 1,000 calls took 15.6 s,
    against 0.05 s with t traced; with the real network each compile costs far more. *Fix:*
    pass t as an array (use `jnp.where` for any branching on it), and loop with `lax.scan` or
    `fori_loop`.

## Minor

11. **`seed=0` is a default.** Every call without a seed returns the same samples, so a caller
    who loops over batches gets duplicates. Make the key a required argument.
12. **The guidance convention isn't documented.** With this formula, w = 0 is unconditional and
    w = 1 conditional. In Ho and Salimans' convention, w = 0 is conditional. Say which one, or
    callers will pick the wrong w.

## Red herrings (say why they're fine)

- **`ALPHA_BAR_PREV = jnp.append(1.0, ALPHA_BAR[:-1])`** is the correct alignment.
- **No noise at the final step** (`if t > 0 else 0.0`) is what DDPM's sampling algorithm does.
- **`ema = params` before the loop:** JAX arrays are immutable, so the alias is harmless (no
  buffer donation here).
- **`% 1.0`** wraps negative coordinates into [0, 1), unlike `jnp.fmod`.
- **`float(loss)` every 1,000 steps** is an occasional host sync, which is fine.
- **The linear β schedule from 1e-4 to 0.02 over 1,000 steps** is the standard DDPM schedule.
- **`key, sub = jax.random.split(key)` in `train`** is correct; the reuse is inside `loss_fn`.

## Model 1-minute summary
"Training and sampling are both broken. In training, three draws share one key, so the lattice
noise copies the coordinate noise, and the condition dropout uses `np.random` inside `jit`, so it
is decided once at trace time and the unconditional model never trains. In sampling, the
guidance halves are swapped, so it pushes away from the requested band gap; the same key is
reused at every step, so the noise accumulates (std 81 instead of 3 in a Gaussian test); and the
posterior variance is used as a standard deviation (samples ten times too tight). After those:
the null condition collides with metals, the coordinate target ignores the wrapping above
σ ≈ 0.2, the EMA is just the current weights, dropout is on while sampling, and the static
timestep recompiles at every step. I'd ask for a 1-D test with Gaussian data and an exact
denoiser, which would have caught every sampler bug here."
