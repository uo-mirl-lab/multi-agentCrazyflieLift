"""Policy rollout + rendering utility used to visualize a trained (or dummy)
policy in the CrazyflieEnv MJX environment."""
import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import mediapy as media

from constants import BASE_HOVER_THRUST


def rollout_policy(
    env,
    params,
    apply_fn,
    episode_length=1500,
    seed=0,
    use_dummy_policy=False,
    use_brax_policy=False,
    print_logs=True,
    random_reset=False,
    target=jnp.array([0.0, 0.0, 1.0]),
):
    """
    Policy rollout function. Rolls out a policy (or a dummy hover policy) in
    `env`, prints per-step diagnostics, and renders the resulting video.

    If use_brax_policy=True, apply_fn is a Brax inference fn
    `apply_fn(obs, rng) -> (action, aux)`; otherwise it's a MAPPO
    Actor.apply, `apply_fn(params, obs_per_agent) -> (mean, log_std)`.
    """
    jit_reset = jax.jit(
        lambda rng, random_reset, target: env.reset(rng, random_reset=random_reset, target=target),
        static_argnames=("random_reset",),
    )
    jit_step = jax.jit(env.step)

    if use_brax_policy:
        jit_inference = jax.jit(apply_fn)
    else:
        def inference_fn(obs_per_agent):
            mean, _ = apply_fn(params, obs_per_agent)
            return mean

        jit_inference = jax.jit(inference_fn)

    # Dummy hover policy action for each drone
    dummy_action_single = jnp.array([BASE_HOVER_THRUST + 0.05, 0.0, 0.0, 0.0])
    dummy_action = jnp.tile(dummy_action_single, (env.num_drones,))

    rng = jax.random.PRNGKey(seed)
    rollout = []

    state = jit_reset(rng, random_reset, target)
    env_target_pos = state.info["target_pos"]
    rollout.append(state)

    print(f"Target: {state.info['target_pos']}")
    print(f"Observation shape for {env.num_drones} drones: {state.obs.shape}")

    total_reward = 0.0

    for i in range(episode_length):
        if bool(state.done):
            break

        if use_dummy_policy:
            action = dummy_action
        elif use_brax_policy:
            act_rng, rng = jax.random.split(rng)
            action, _ = jit_inference(state.obs, act_rng)
        else:
            per_agent_obs_dim = state.obs.shape[0] // env.num_drones
            obs = state.obs.reshape((env.num_drones, per_agent_obs_dim))
            action = jit_inference(obs).reshape((-1,))

        # Env step
        state = jit_step(state, action)
        rollout.append(state)
        total_reward += float(state.reward)

        # Logs
        obs_np = np.asarray(state.obs)
        pos = obs_np[0:3]
        lin_vel = obs_np[3:6]
        ang_vel = obs_np[6:9]

        num_other_drone_pos = env.num_drones * 3
        oldest_action = obs_np[24 + num_other_drone_pos: 28 + num_other_drone_pos]
        action_np = np.asarray(action)

        if print_logs:
            print(f"\n\nStep {i:04d}, Action {action_np}, Reward {float(state.reward):+.4f}")

            print(f"  \nPartial observation for drone 1")
            print(f"    Pos:          {pos}")
            print(f"    Linear vel:   {lin_vel}")
            print(f"    Angular vel:  {ang_vel}")
            print(f"    Oldest Action in history:  {oldest_action}")

            metrics_np = jax.tree_util.tree_map(
                lambda x: np.asarray(x).item() if np.size(x) == 1 else np.asarray(x),
                state.metrics
            )
            print(f"  \nReward breakdown")
            print(f"    Dist Improvement: {metrics_np.get('reward/distance_improvement', np.nan):+.4f}")
            print(f"    Target Proximity Bonus: {metrics_np.get('reward/proximity_bonus', np.nan):+.4f}")
            print(f"    Survival Bonus: {metrics_np.get('reward/survival_bonus', np.nan):+.4f}")
            print(f"    Drone Proximity Penalty: {metrics_np.get('reward/drone_proximity_penalty', np.nan):+.4f}")
            print(f"    Termination: {metrics_np.get('reward/termination', np.nan):+.4f}")

    # Summary
    rewards = [float(s.reward) for s in rollout]
    print(f"\nTotal steps: {len(rewards)}, Final reward: {total_reward:.2f}")

    # Utility to render target as geom in scene
    print(f"\nRendering trajectory, target at {env_target_pos}...")
    target_geom_id = mujoco.mj_name2id(env.mj_model, mujoco.mjtObj.mjOBJ_GEOM, b"target")

    def move_target_geom(scene: mujoco.MjvScene, target_pos: np.ndarray):
        scene.geoms[target_geom_id].pos[:] = target_pos

    modify_target_fn = lambda scene: move_target_geom(scene, env_target_pos)

    # Collect frames, use track camera from xml scene and update geom pos to target pos
    frames = env.render(
        rollout,
        camera="track",
        modify_scene_fns=[modify_target_fn] * len(rollout),
    )
    media.show_video(frames, fps=1.0 / env.dt)

    return rollout
