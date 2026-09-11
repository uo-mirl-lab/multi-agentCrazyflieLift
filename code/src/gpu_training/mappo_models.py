"""
Flax model definitions and PPO/MAPPO math helpers: the decentralized
Gaussian policy (Actor), centralized value function (CentralizedCritic), the
pytree train-state wrapper, and small numeric helpers (log-prob, gradient
clipping).

Extracted from the "MAPPO Training setup" cell of MARL_Crazyflie.ipynb.
"""
import functools
from dataclasses import dataclass
from typing import Tuple

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training.train_state import TrainState


class MLP(nn.Module):
    """Simple feed-forward MLP with ReLU activations."""
    hidden_sizes: Tuple[int, ...]
    activate_final: bool = False

    @nn.compact
    def __call__(self, x):
        for h in self.hidden_sizes:
            x = nn.relu(nn.Dense(h)(x))
        if self.activate_final:
            x = nn.relu(x)
        return x


class Actor(nn.Module):
    """Per-agent Gaussian policy: state -> (mean, log_std)."""
    hidden_sizes: Tuple[int, ...]
    action_dim: int

    @nn.compact
    def __call__(self, x):
        # x: (..., obs_dim_per_agent)
        h = MLP(self.hidden_sizes)(x)

        # Layers for mean and log_std weights & biases
        mean = nn.Dense(self.action_dim)(h)
        log_std = nn.Dense(self.action_dim)(h)

        # Ensure numerically reasonable range for log_std
        log_std = jnp.clip(log_std, -20.0, 2.0)

        return mean, log_std


@functools.partial(jax.jit, static_argnums=(1,))
def actor_forward(params, apply_fn, per_agent_obs):
    """
    Decentralized (per-agent) policy forward pass, sharing parameters
    across agents.

    Parameters
    ----------
    params : Any
        Actor network parameters.
    apply_fn : Callable
        Actor.apply (static for jit).
    per_agent_obs : jax.Array
        (batch, n_agents, per_agent_obs_dim)
    """
    B, A, D = per_agent_obs.shape
    flat = per_agent_obs.reshape((B * A, D))

    mean, log_std = apply_fn(params, flat)  # shapes (B*A, action_dim)
    mean = mean.reshape((B, A, -1))
    log_std = log_std.reshape((B, A, -1))

    return mean, log_std


class CentralizedCritic(nn.Module):
    """Takes the global joint observation (all agents concatenated) -> scalar value."""
    hidden_sizes: Tuple[int, ...]

    @nn.compact
    def __call__(self, x):
        h = MLP(self.hidden_sizes)(x)
        v = nn.Dense(1)(h)
        return jnp.squeeze(v, -1)  # (batch,)


@functools.partial(jax.jit, static_argnums=(1,))
def critic_forward(params, apply_fn, global_obs):
    """
    Centralized critic forward pass: sees all agents' observations and
    computes a single centralized value estimate.
    """
    return apply_fn(params, global_obs)  # (batch,)


# TrainState wrapper, registered as a pytree so it can flow through jax.lax.scan.
# https://flax.readthedocs.io/en/latest/_modules/flax/training/train_state.html
@jax.tree_util.register_pytree_node_class
@dataclass
class PPOTrainState:
    """Bundles the actor/critic TrainStates with per-env rollout bookkeeping."""
    policy_state: TrainState
    value_state: TrainState
    env_steps: jnp.ndarray   # shape (num_envs,), int32
    ep_returns: jnp.ndarray  # shape (num_envs,)

    def tree_flatten(self):
        children = (self.policy_state, self.value_state, self.env_steps, self.ep_returns)
        return children, None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


def gaussian_log_prob(mean, log_std, action):
    """Log-probability of `action` under a diagonal Gaussian(mean, exp(log_std))."""
    # mean, log_std, action shapes: (..., action_dim)
    var = jnp.exp(2.0 * log_std)
    logp = -0.5 * (((action - mean) ** 2) / (var + 1e-8) + 2 * log_std + jnp.log(2 * jnp.pi))
    return jnp.sum(logp, axis=-1)  # sum over action dim


def clip_by_global_norm(updates, max_norm):
    """Safe scalar clipping of a pytree of updates by their global norm (for jax.grad stability)."""
    g_norm = jnp.sqrt(sum(jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(updates)))
    trigger = g_norm < max_norm
    coef = jnp.where(trigger, 1.0, max_norm / (g_norm + 1e-6))
    return jax.tree_util.tree_map(lambda p: p * coef, updates)


def make_deterministic_policy(apply_fn, params):
    """
    Wrap an Actor's apply_fn/params into a jitted, deterministic
    (mean-action) policy function taking per-agent observations and
    returning the mean action.

    Mirrors the `inference_fn` pattern used in the notebook both for the
    PPO test rollout and when loading a saved MAPPO checkpoint.
    """
    def inference_fn(obs_per_agent):
        mean, _ = apply_fn(params, obs_per_agent)
        return mean

    return jax.jit(inference_fn)
