"""
PR #281: Band-gap-conditioned crystal generation with classifier-free guidance (JAX)

Extends our DiffCSP-style model, which generates the lattice and fractional coordinates for a
given composition, with a band-gap condition trained by classifier-free guidance: the
condition is dropped for 10% of training steps, and the sampler mixes the conditional and
unconditional predictions with guidance weight w. The lattice follows a DDPM (epsilon
prediction); the coordinates follow a variance-exploding process on the torus (score
prediction), as in DiffCSP. Sampling uses the EMA weights.
"""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax

from model import denoiser  # denoiser(params, frac, lat, t, gap, train) -> (score_frac, eps_lat)

T = 1000
BETAS = jnp.linspace(1e-4, 0.02, T)
ALPHAS = 1.0 - BETAS
ALPHA_BAR = jnp.cumprod(ALPHAS)
ALPHA_BAR_PREV = jnp.append(1.0, ALPHA_BAR[:-1])
POST_VAR = BETAS * (1.0 - ALPHA_BAR_PREV) / (1.0 - ALPHA_BAR)
SIGMAS = jnp.exp(jnp.linspace(jnp.log(0.005), jnp.log(0.5), T))  # coordinate noise levels

P_UNCOND = 0.1
NULL_GAP = 0.0
EMA_DECAY = 0.999
OPT = optax.adamw(optax.warmup_cosine_decay_schedule(0.0, 3e-4, 1_000, 200_000))


def loss_fn(params, batch, key, drop):
    frac0, lat0, gap = batch["frac"], batch["lattice"], batch["gap"]
    t = jax.random.randint(key, (frac0.shape[0],), 0, T)
    eps_f = jax.random.normal(key, frac0.shape)
    eps_l = jax.random.normal(key, lat0.shape)
    ab, sigma = ALPHA_BAR[t][:, None, None], SIGMAS[t][:, None, None]
    lat_t = jnp.sqrt(ab) * lat0 + jnp.sqrt(1.0 - ab) * eps_l
    frac_t = (frac0 + sigma * eps_f) % 1.0
    cond = jnp.where(drop, NULL_GAP, gap)
    score_f, pred_l = denoiser(params, frac_t, lat_t, t, cond, train=True)
    target_f = -eps_f / sigma  # score of the Gaussian perturbation
    return jnp.mean((pred_l - eps_l) ** 2) + jnp.mean(sigma**2 * (score_f - target_f) ** 2)


@jax.jit
def train_step(params, opt_state, batch, key):
    drop = np.random.rand() < P_UNCOND
    loss, grads = jax.value_and_grad(loss_fn)(params, batch, key, drop)
    updates, opt_state = OPT.update(grads, opt_state, params)
    return optax.apply_updates(params, updates), opt_state, loss


def train(params, loader, n_steps, seed=0):
    key = jax.random.PRNGKey(seed)
    opt_state = OPT.init(params)
    ema = params
    for step, batch in zip(range(n_steps), loader):
        key, sub = jax.random.split(key)
        params, opt_state, loss = train_step(params, opt_state, batch, sub)
        ema = optax.incremental_update(params, ema, step_size=EMA_DECAY)
        if step % 1000 == 0:
            print(f"step {step} loss {float(loss):.4f}")
    return params, ema


@partial(jax.jit, static_argnums=(3,))
def guided(params, frac, lat, t, gap, w):
    cond = jnp.concatenate([jnp.full_like(gap, NULL_GAP), gap])
    tt = jnp.full(2 * gap.shape[0], t)
    score, eps = denoiser(params, jnp.concatenate([frac, frac]), jnp.concatenate([lat, lat]),
                          tt, cond, train=True)
    (s_c, s_u), (e_c, e_u) = jnp.split(score, 2), jnp.split(eps, 2)
    return s_u + w * (s_c - s_u), e_u + w * (e_c - e_u)


def sample(ema_params, n_atoms, gap, w=2.0, seed=0):
    key = jax.random.PRNGKey(seed)
    k_f, k_l = jax.random.split(key)
    frac = jax.random.uniform(k_f, (gap.shape[0], n_atoms, 3))
    lat = jax.random.normal(k_l, (gap.shape[0], 3, 3))
    for t in range(T - 1, -1, -1):
        score, eps = guided(ema_params, frac, lat, t, gap, w)
        # lattice: DDPM reverse step
        mean = (lat - BETAS[t] / jnp.sqrt(1.0 - ALPHA_BAR[t]) * eps) / jnp.sqrt(ALPHAS[t])
        z = jax.random.normal(key, lat.shape) if t > 0 else 0.0
        lat = mean + POST_VAR[t] * z
        # coordinates: reverse-SDE step on the torus
        step = SIGMAS[t] ** 2 - (SIGMAS[t - 1] ** 2 if t > 0 else 0.0)
        frac = (frac + step * score + jnp.sqrt(step) * jax.random.normal(key, frac.shape)) % 1.0
    return frac, lat
