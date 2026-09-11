"""Vectorized (multi-environment) reset/step helpers used by both the custom
MAPPO trainer and target/seed evaluation, plus a helper to construct an env
and infer its observation/action dimensions."""
import functools

import jax


def make_env_and_infer(env_factory):
    """Build an environment via `env_factory()` and infer observation/action
    dimensions from a single reset."""
    env = env_factory()
    key = jax.random.PRNGKey(0)
    st = env.reset(key)
    obs = st.obs

    # obs is flattened across drones: shape (per_env_obs_dim,)
    per_env_obs_dim = obs.shape[0]
    action_dim = env.action_size  # total action size (flat, across drones)

    # Per-agent obs dimension
    per_agent_obs_dim = per_env_obs_dim // env.num_drones

    return env, per_env_obs_dim, per_agent_obs_dim, action_dim


@functools.partial(jax.jit, static_argnums=0)
def vec_reset(env, rngs):
    """Reset env(s) based on rng key(s), one per environment."""
    return jax.vmap(lambda k: env.reset(k))(rngs)


@functools.partial(jax.jit, static_argnums=0)
def vec_step(env, states, actions):
    """Step env(s). `states` is a pytree of length num_envs, `actions` is
    flattened (num_envs, action_dim)."""
    return jax.vmap(lambda s, a: env.step(s, a))(states, actions)


@functools.partial(jax.jit, static_argnums=0)
def vec_reset_targets(env, rngs, targets):
    """Reset env(s) with fixed (non-random) per-env targets."""
    return jax.vmap(lambda rng, target: env.reset(rng, random_reset=False, target=target))(rngs, targets)
