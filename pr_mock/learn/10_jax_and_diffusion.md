# 10. JAX and diffusion models

JAX code rarely crashes when it is wrong. Random numbers come from keys that you split by hand,
so a reused key quietly gives the same noise twice. `jit` runs the Python body once, while
tracing, so NumPy draws, globals and list appends are frozen into the compiled function. Arrays
are immutable, float64 is off by default, and out-of-range indices clamp instead of raising.

Diffusion samplers must agree with the training code on the parameterisation, the schedule, the
variance, the guidance convention and the geometry of fractional coordinates. A mismatch still
gives plausible structures. The 1-D tests below use Gaussian data (std 3) and an exact denoiser.

Outputs come from Python 3.14.5 on CPU with JAX 0.12.0.1, optax 0.2.8 and Flax 0.12.12, plus
Equinox 0.13.8 and diffusers 0.42.1 for one line each. Timings vary by machine.

## Quick table

| You see | What goes wrong | Fix |
|---|---|---|
| one `key` reused for several draws, for every sample (`vmap` with `in_axes=None`) or for every step (`scan`) | identical or dependent noise | `jax.random.split`: one subkey per draw, per sample and per step |
| `np.random`, a global key, `print` or `list.append` inside `jit` | runs once, at trace time | pass keys in; return values; `jax.debug.print` |
| `jnp.asarray` on float64 data | silent float32 | `jax_enable_x64` at start-up; assert dtypes |
| `static_argnums` on a timestep, or a new atom count in every batch | one compilation per value or shape | keep it traced (`jnp.where`, not `if`); pad to a few sizes and mask |
| a Python `for` over sampler steps inside `jit` | compile time grows with the step count | `lax.scan` or `lax.fori_loop` |
| `time.perf_counter()` around a JAX call | measures dispatch, not work | warm up, then `.block_until_ready()` |
| `x.at[i].set(v)` or `eqx.nn.inference_mode(m)` with the result unused | nothing changes | assign the result |
| `opt.init(params)` inside the training loop | Adam moments and the schedule reset every step | create it once; keep it in the train state |
| `optax.incremental_update(new, old, step_size=decay)` | the "EMA" is the latest weights | `step_size=1 - decay` |
| `jnp.where(mask, 1 / r, 0.0)` where masked `r` is 0 | NaN gradients behind a finite forward pass | mask the input too: `r = jnp.sqrt(jnp.where(mask, r2, 1.0))` |
| an index, `one_hot` class or label out of range | clamped, a zero row or NaN; no error | assert ranges outside `jit` |
| `jnp.fmod(frac, 1.0)` on fractional coordinates | negative values stay negative | `jnp.mod` |
| `jnp.clip(x0_hat, -1, 1)` or `clip_sample=True` on coordinates | samples squashed into [-1, 1] | clip only data that lives in [-1, 1] |

## Examples

### Reused keys

```python
def corrupt(key, x, lattice):
    t = jax.random.uniform(key, (x.shape[0],))
    eps_x = jax.random.normal(key, x.shape)
    eps_l = jax.random.normal(key, lattice.shape)
    return t, eps_x, eps_l
# x (64, 8, 3), lattice (64, 3, 3): eps_l == eps_x.ravel()[:576] exactly
def body(x, t):                          # the step of a lax.scan sampler
    z = jax.random.normal(key, x.shape)
    return posterior_mean(x, t) + sigma[t] * z, None
# 1000 steps on data with std 3.0: samples end with std 81.8
```

A key gives the same bits every time: `t` equals `norm.cdf(eps_x.ravel()[:64])` to within
8.9e-08, and the sampler adds one z a thousand times. `vmap` with `in_axes=None` for the key gives
every sample the same noise. **Fix:** split per draw, per sample and per step; carry the key.

### Python state frozen at trace time

```python
key, losses = jax.random.key(0), []
@jax.jit
def train_step(cond):
    t = jax.random.randint(key, (4,), 0, 1000)
    if np.random.rand() < 0.1:              # drop the condition for CFG
        cond = jnp.zeros_like(cond)
    losses.append(cond.sum())
    return t, cond
# 500 calls: 1 distinct t; the condition dropped in 0 of 500 steps (another seed: 500 of 500)
```

The body runs once, while tracing: the global key is a constant (reassigning it later changes
nothing), and `float(losses[0])` raises `ConcretizationTypeError`. **Fix:** pass a key in and
split it; drop the condition per sample with `jax.random.bernoulli`; return values to log them.

### float64 becomes float32

```python
energies = np.array([-1234.567891, -1234.567812])   # eV, from a DFT table
e = jnp.asarray(energies)
e.dtype                      # float32, with no warning
float(e[1] - e[0])           # 0.0; NumPy gives 7.9e-05
```

Only an explicit `dtype=jnp.float64` warns ("UserWarning: Explicitly requested dtype float64
... will be truncated to dtype float32"). **Fix:** `jax.config.update("jax_enable_x64", True)`
at start-up, since arrays made earlier stay float32, and assert dtypes where data enters.

### Recompiling on every call

```python
@partial(jax.jit, static_argnums=2)    # added because "if t > 1" raised TracerBoolConversionError
def denoise_step(params, x, t):
    x = x - 0.01 * net(params, x, t)
    if t > 1:
        x = x + sigma[t] * noise
    return x
for t in range(200, 0, -1):
    x = denoise_step(params, x, t)     # 200 traces, 7 to 8 s; with t traced: 1 trace, 0.05 s
```

Shapes act the same way: 100 batches with 45 different atom counts compiled 45 times, and
padding to the next power of two cut that to 3. **Fix:** keep `t` traced with
`jnp.where(t > 1, sigma[t], 0.0)`; pad to a few sizes and mask padded atoms in every reduction.

### Loops inside `jit`, and timing

```python
@partial(jax.jit, static_argnums=1)
def sample(x, n_steps):
    for t in range(n_steps, 0, -1):
        x = denoise(x, t)
    return x
sample(x_T, 1000)                    # 21 to 22 s, nearly all of it compilation
t0 = time.perf_counter()
x0 = sample(x_T, 1000)
print(time.perf_counter() - t0)      # about 5e-05
```

The loop is unrolled into 1000 copies of the network; with `lax.scan` the first call took about
0.1 s. The timer stops before the work does: blocking on `x0` first, the call took 53 to 61 ms.
**Fix:** `jax.lax.scan` over the steps; time a warmed-up call with `.block_until_ready()`.

### Discarded updates

```python
pos.at[pad_idx].set(0.0)               # returns a new array
float(pos.sum())                       # 12.0: unchanged
eqx.nn.inference_mode(model)           # returns a new model
a, b = model(x, key=jax.random.key(1)), model(x, key=jax.random.key(2))
jnp.allclose(a, b)                     # False: dropout is still on
```

JAX arrays and Equinox modules are immutable, so both calls return a result and change nothing.
**Fix:** `pos = pos.at[pad_idx].set(0.0)` and `model = eqx.nn.inference_mode(model)`.

### Optimiser state and EMA

```python
opt = optax.adamw(optax.warmup_cosine_decay_schedule(0.0, 3e-4, 1000, 100_000))
for batch in batches:
    opt_state = opt.init(params)
    grads = jax.grad(loss_fn)(params, batch)
    updates, opt_state = opt.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    ema = optax.incremental_update(params, ema, step_size=0.999)
# after 200 steps, params equal their initial values
```

A fresh state is at step 0, where the warm-up gives a learning rate of 0.0. `incremental_update`
returns `step_size * new + (1 - step_size) * old`, so this "EMA" follows the latest weights (new
1.0, old 0.0 gives 0.999). **Fix:** create `opt_state` once and carry it; `step_size=1 - decay`.

### Sampler formulas

```python
# the model was trained to predict v = sqrt(ab[t]) * eps - sqrt(1 - ab[t]) * x0
x0_hat = (x_t - jnp.sqrt(1 - ab[t]) * model(x_t, t)) / jnp.sqrt(ab[t])
ab_prev = jnp.append(ab[1:], ab[-1])
coef_x0 = jnp.sqrt(ab_prev) * betas / (1 - ab)
coef_xt = jnp.sqrt(alphas) * (1 - ab_prev) / (1 - ab)
post_var = betas * (1 - ab_prev) / (1 - ab)
x = coef_x0[t] * x0_hat + coef_xt[t] * x_t + post_var[t] * z
```

Three bugs: the ε formula on a v-model, `ab_prev` shifted the wrong way, and a variance used as
a standard deviation. In the 1-D test, each alone gives std 188587, 5.2e7 and 0.296; the fixed
code gives 3.00. **Fix:** for a v-model, x0 = sqrt(ab) * x_t - sqrt(1 - ab) * v;
`jnp.append(1.0, ab[:-1])`; `jnp.sqrt(post_var[t]) * z`; one `prediction_type` for both sides.

### Fractional coordinates on the torus

```python
frac = jnp.fmod(frac + delta, 1.0)     # [-0.3, 0.2]; jnp.mod gives [0.7, 0.2]
x_t = (x0 + sigma * eps) % 1.0
target = -eps / sigma                  # 55% RMS error against the true score at sigma = 0.2
velocity = x1 - x0                     # the long way round for 25% of uniform pairs
```

The wrapped noise has a wrapped-normal score, and `-eps / sigma` drifts from it as σ grows:
negligible at 0.1, 194% at 0.3, 3126% at 0.5. In float32, `-1e-8 % 1.0` is exactly 1.0.
**Fix:** `jnp.mod`; a target summed over periodic images; velocity `(x1 - x0 + 0.5) % 1.0 - 0.5`.

### Classifier-free guidance

```python
out = model(jnp.concatenate([x, x]), jnp.concatenate([null_cond, cond]))
eps_c, eps_u = jnp.split(out, 2)
eps = eps_u + w * (eps_c - eps_u)
# eps_c = 1.0, eps_u = 0.2, w = 2: gives -0.6 instead of 1.8
```

The halves are swapped, so guidance pushes away from the condition. With w + 1 in place of w,
this form equals Ho and Salimans' `(1 + w) * eps_c - w * eps_u` (2.6 for both here); w = 0 is
unconditional here and plain conditional there. **Fix:** split by name; use a learned null
embedding, never a real value such as a band gap of 0.0 eV.

### Schedule endpoints and clipping

```python
T = 100
betas = jnp.linspace(1e-4, 0.02, T)           # the range used with T = 1000
jnp.sqrt(jnp.cumprod(1 - betas)[-1])          # 0.603: x_T keeps 60% of the signal
x0_hat = jnp.clip(x0_hat, -1.0, 1.0)          # copied from image code
# 1-D test: std 0.768 and max |x| 1.000; without the clip, std 3.00
```

With T = 1000 the same range ends at 0.006. Without its 0.999 clip, a cosine schedule ends at
ᾱ_T = 0.0. diffusers' `DDPMScheduler` and `DDIMScheduler` default to `clip_sample=True`. A flow
Euler loop over all of `jnp.linspace(0, 1, N)` divides by 1 - t = 0, so all samples are NaN or
inf. **Fix:** check √ᾱ_T; clip betas; don't clip coordinates; step over `zip(ts[:-1], ts[1:])`.

## Looks wrong, is fine

- `sigma_t**2 = betas[t]` instead of the posterior variance. DDPM allows both; the 1-D test gives
  std 3.001 and 2.998.
- Timesteps from `randint(key, (B,), 1, T + 1)` and loops over `range(T, 0, -1)`. Correct when
  the schedule arrays have T + 1 entries with a leading 0. Check the array length.
- Subtracting the mean over atoms from predicted velocities or scores on the torus. They live in
  a flat tangent space. Doing it to wrapped positions is wrong: 0.05 and 0.95 average to 0.5.
- `vmap` over `jax.random.split(key, B)` differs from one `jax.random.normal(key, (B,))`; both are
  fine. Version-dependent: values also depend on `jax_threefry_partitionable`, on since JAX 0.5.0.
- Flax NNX `model.eval()` with its result unused. Unlike Equinox it changes the model in place
  and returns `None`. Version-dependent: this is the NNX API; Linen takes a `train` argument.
