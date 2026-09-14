"""
Apples-to-apples comparison between a frozen low-level PPO hover policy run
"as-is" (no correction) and the same policy run underneath a trained
translator (cf_translator_env.CrazyflieTranslatorEnv), across many
independently-randomized "sim-to-real gap" episodes.

Both policies are rolled out through the same env instance with the same
per-seed reset keys, so each seed samples identical physics for both -- the
only difference is whether anything corrects for it. Mirrors evaluation.py's
batched-rollout style, adapted to the translator's per-channel-scale action.
"""
import jax
import jax.numpy as jnp
from tabulate import tabulate

from vec_env_utils import vec_reset, vec_step


def identity_scale_policy(obs, rng):
    """The "no correction" baseline: always scales the low-level policy's raw
    action by 1.0. Same (obs, rng) -> (action, extras) signature as a Brax
    inference_fn, so it's a drop-in for `translator_fn` below. Returns
    raw action 0 (not scale 1) -- CrazyflieTranslatorEnv.step() maps a raw
    action of 0 to a scale of 1.0 (see cf_translator_env._remap_scale)."""
    del obs, rng
    return jnp.zeros(4), None


def rollout_batch(env, apply_fn, num_seeds, episode_length, base_seed=0, target=None):
    """Roll out `apply_fn` (Brax-style `(obs, rng) -> (action, extras)`)
    against `num_seeds` parallel, independently randomized instances of
    `env`. Call with the same base_seed to compare two policies under
    identical per-seed conditions. If `target` is given every seed resets to
    it (random_reset=False); otherwise each seed gets its own randomized
    start/target."""
    rng = jax.random.PRNGKey(base_seed)
    rng, *reset_keys = jax.random.split(rng, num_seeds + 1)
    reset_keys = jnp.stack(reset_keys)

    if target is None:
        states = vec_reset(env, reset_keys)
    else:
        targets = jnp.stack([target] * num_seeds)
        states = jax.vmap(lambda k, t: env.reset(k, random_reset=False, target=t))(reset_keys, targets)

    jit_apply = jax.jit(jax.vmap(apply_fn))

    total_rewards = jnp.zeros(num_seeds)
    steps = jnp.zeros(num_seeds, dtype=int)
    done_mask = jnp.zeros(num_seeds, dtype=bool)

    act_rng = jax.random.PRNGKey(base_seed + 1)

    for _ in range(episode_length):
        if bool(jnp.all(done_mask)):
            break

        act_rng, step_key = jax.random.split(act_rng)
        step_keys = jax.random.split(step_key, num_seeds)

        actions, _ = jit_apply(states.obs, step_keys)
        states = vec_step(env, states, actions)

        rewards = states.reward
        dones = states.done

        total_rewards = total_rewards + rewards * (~done_mask)
        steps = steps + (~done_mask).astype(int)
        done_mask = jnp.logical_or(done_mask, dones.astype(bool))

    return total_rewards, steps


def compare_policies(env, low_level_only_fn, translator_fn, num_seeds, episode_length, base_seed=0, target=None):
    """
    Roll both policies out against identical randomized physics and return a
    dict of {"low_level_only": {...}, "with_translator": {...}} summary stats.
    """
    results = {}
    for name, apply_fn in (("low_level_only", low_level_only_fn), ("with_translator", translator_fn)):
        total_rewards, steps = rollout_batch(
            env, apply_fn, num_seeds=num_seeds, episode_length=episode_length,
            base_seed=base_seed, target=target,
        )
        results[name] = {
            "total_rewards": total_rewards,
            "steps": steps,
            "mean_reward": float(jnp.mean(total_rewards)),
            "std_reward": float(jnp.std(total_rewards)),
            "mean_steps": float(jnp.mean(steps)),
            "success_rate": float(jnp.mean(steps >= episode_length)),
        }
    return results


def print_comparison_table(results, episode_length):
    """Pretty-print the dict returned by compare_policies(...)."""
    headers = ["Policy", "Mean Reward", "Std Reward", "Mean Steps", f"Success Rate (steps=={episode_length})"]
    rows = []
    for name, r in results.items():
        rows.append([
            name,
            f"{r['mean_reward']:.2f}",
            f"{r['std_reward']:.2f}",
            f"{r['mean_steps']:.1f}",
            f"{r['success_rate'] * 100:.1f}%",
        ])
    print(tabulate(rows, headers=headers, tablefmt="github"))
