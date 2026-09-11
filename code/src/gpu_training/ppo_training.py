"""
Helpers for configuring and launching Brax PPO training (the single-agent
baseline trained against a single-drone CrazyflieEnv).

Extracted from the "PPO Training" cell of MARL_Crazyflie.ipynb.
"""
import functools

from ml_collections import config_dict
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo

from training_plots import linear_annealing_lr


def default_ppo_params(num_timesteps=60_000_000, learning_rate=None, **overrides):
    """
    Build the Brax PPO training config used in the notebook.

    Parameters
    ----------
    num_timesteps : int
    learning_rate : float or Callable, optional
        Defaults to a linear anneal from 3e-3 to 0.25 * 3e-3 over
        `num_timesteps` (see `training_plots.linear_annealing_lr`).
    **overrides
        Any other `config_dict.create(...)` kwargs to override the defaults
        (e.g. `num_envs=1024`).
    """
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

    Parameters
    ----------
    ppo_params : config_dict.ConfigDict
        As returned by `default_ppo_params(...)`.
    progress_fn : Callable
        e.g. a `training_plots.PPOProgressPlotter` instance.
    policy_hidden, value_hidden : tuple of int
        Hidden layer sizes for the PPO actor/value networks.

    Notes
    -----
    This mirrors the original notebook cell's logic exactly, including a
    quirk: the custom-hidden-size `network_factory` partial below is only
    actually used if `"network_factory"` is itself a key inside
    `ppo_params` (it isn't, by default) — otherwise the stock Brax network
    factory is used and `policy_hidden`/`value_hidden` are silently
    ignored. Left as-is rather than "fixed" during extraction; worth a
    second look if you want policy/value hidden sizes to actually apply.
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
