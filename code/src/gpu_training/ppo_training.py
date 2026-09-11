"""Helpers for configuring and launching Brax PPO training (the single-agent
baseline trained against a single-drone CrazyflieEnv)."""
import functools

import jax
from ml_collections import config_dict
from brax.io import model as brax_model
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo

from training_plots import linear_annealing_lr


def default_ppo_params(num_timesteps=60_000_000, learning_rate=None, **overrides):
    """Build the Brax PPO training config used in the notebook. `learning_rate`
    defaults to a linear anneal from 3e-3 to 0.25 * 3e-3 over
    `num_timesteps`; **overrides can override any other config_dict field
    (e.g. `num_envs=1024`)."""
    if learning_rate is None:
        learning_rate = linear_annealing_lr(3e-3, 0.25 * 3e-3, num_timesteps)

    params = dict(
        num_timesteps=num_timesteps,
        num_evals=10,
        episode_length=1500,
        normalize_observations=True,
        action_repeat=1,
        unroll_length=30,
        num_minibatches=32,
        num_updates_per_batch=16,
        discounting=0.995,
        learning_rate=learning_rate,
        entropy_cost=1e-3,
        num_envs=2048,
        batch_size=1024,
    )
    params.update(overrides)
    return config_dict.create(**params)


def build_ppo_train_fn(ppo_params, progress_fn=lambda *args: None, policy_hidden=(128, 128), value_hidden=(128, 128)):
    """
    Build a `functools.partial(ppo.train, ...)`, ready to call with
    `environment=...` and `wrap_env_fn=mujoco_playground.wrapper.wrap_for_brax_training`.

    Note: the custom-hidden-size `network_factory` below is only actually
    used if `"network_factory"` is itself a key inside `ppo_params` (it
    isn't, by default) -- otherwise the stock Brax network factory is used
    and `policy_hidden`/`value_hidden` are silently ignored. Worth a look if
    you want the hidden sizes to actually apply.
    """
    training_params = dict(ppo_params)

    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=policy_hidden,
        value_hidden_layer_sizes=value_hidden,
    )
    network_factory = ppo_networks.make_ppo_networks
    if "network_factory" in ppo_params:
        del training_params["network_factory"]
        network_factory = functools.partial(
            ppo_networks.make_ppo_networks,
            **ppo_params.network_factory
        )

    return functools.partial(
        ppo.train,
        **training_params,
        network_factory=network_factory,
        progress_fn=progress_fn,
    )


def load_ppo_policy(checkpoint_path, env, normalize_observations=True):
    """
    Reconstruct a frozen `make_inference_fn` from a checkpoint saved via
    `brax.io.model.save_params` (e.g. constants.PPO_MODEL_SAVE_PATH), instead
    of training one. Returns (make_inference_fn, params), same as
    build_ppo_train_fn(...)'s train_fn -- so `make_inference_fn(params,
    deterministic=True)` works the same either way.

    Rebuilds the same stock-hidden-size network build_ppo_train_fn actually
    trains (see its docstring note), since the network architecture isn't
    itself part of the saved checkpoint.
    """
    obs_size = env.reset(jax.random.PRNGKey(0)).obs.shape[-1]
    normalize_fn = running_statistics.normalize if normalize_observations else (lambda x, y: x)
    network = ppo_networks.make_ppo_networks(obs_size, env.action_size, preprocess_observations_fn=normalize_fn)
    make_inference_fn = ppo_networks.make_inference_fn(network)
    params = brax_model.load_params(checkpoint_path)
    return make_inference_fn, params
