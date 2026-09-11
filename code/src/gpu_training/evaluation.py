"""Batched target/seed evaluation of a trained (MAPPO) policy: run many
(target, seed) rollouts in parallel and report mean return / survival
stats in a table."""
import jax
import jax.numpy as jnp
from tabulate import tabulate

from vec_env_utils import vec_reset_targets, vec_step


def test_targets_batched(env, jit_inference, num_drones, per_agent_obs_dim, targets, keys, episode_length=1500):
    """Run one rollout per (target, key) pair in parallel. jit_inference is a
    deterministic policy fn `(per_agent_obs) -> flat_action_per_agent`, e.g.
    from `mappo_models.make_deterministic_policy(...)`. Returns per-environment
    cumulative reward and steps survived."""
    num_envs = len(targets)

    states = vec_reset_targets(env, keys, targets)

    total_rewards = jnp.zeros(num_envs, dtype=float)
    steps = jnp.zeros(num_envs, dtype=int)
    done_mask = jnp.zeros(num_envs, dtype=bool)

    for t in range(episode_length):
        if bool(jnp.all(done_mask)):
            break

        per_agent_obs = states.obs.reshape((num_envs, num_drones, per_agent_obs_dim))
        actions = jit_inference(per_agent_obs).reshape((num_envs, -1))

        states = vec_step(env, states, actions)
        rewards = states.reward.reshape(num_envs)
        dones = states.done.reshape(num_envs)

        # Update steps and rewards while not done
        total_rewards += rewards * (~done_mask)
        steps = steps + (~done_mask)
        done_mask = jnp.logical_or(done_mask, dones)

    return total_rewards, steps


def make_target_seed_batch(targets, num_seeds, base_seed=0):
    """Repeat each target `num_seeds` times and generate a matching PRNG key per (target, seed)."""
    num_targets = len(targets)

    targets_batched = jnp.repeat(jnp.stack(targets), num_seeds, axis=0)

    rng = jax.random.PRNGKey(base_seed)
    rng, *keys = jax.random.split(rng, num_targets * num_seeds + 1)
    keys = jnp.stack(keys)

    return targets_batched, keys, num_targets, num_seeds


def eval_targets_with_seeds(env, jit_inference, num_drones, per_agent_obs_dim, targets, num_seeds, episode_length, base_seed=0):
    """
    Evaluate `jit_inference` on every target in `targets`, each repeated
    over `num_seeds` random seeds, and reshape results to [targets, seeds].
    """
    targets_batched, keys, T, S = make_target_seed_batch(targets, num_seeds, base_seed)

    returns, steps = test_targets_batched(
        env, jit_inference, num_drones, per_agent_obs_dim,
        targets_batched, keys=keys, episode_length=episode_length,
    )

    returns = returns.reshape(T, S)
    steps = steps.reshape(T, S)

    return returns, steps


def print_results_table(targets, returns, steps, episode_length):
    """Print a table of mean/std return, mean steps survived, and failure counts per target."""
    stats = {
        "mean_return": returns.mean(axis=1),
        "std_return": returns.std(axis=1),
        "mean_steps": steps.mean(axis=1),
        "failures": (steps < episode_length).sum(axis=1),
    }

    rows = []
    for i, target in enumerate(targets):
        rows.append([
            i,
            "[" + ", ".join(f"{x:.1f}" for x in target) + "]",
            f"{stats['mean_return'][i]:.2f}",
            f"{stats['std_return'][i]:.2f}",
            f"{stats['mean_steps'][i]:.1f}",
            f"{int(stats['failures'][i])}/{steps.shape[1]}",
        ])

    headers = [
        "Target #",
        "Target (x,y,z)",
        "Mean Return",
        "Std Return",
        "Mean Steps",
        "Failures",
    ]

    print(tabulate(rows, headers=headers, tablefmt="github"))
