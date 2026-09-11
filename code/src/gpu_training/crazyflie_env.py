"""
CrazyflieEnv: MJX (MuJoCo playground) environment for the Crazyflie drone,
supporting multiple drones and MARL rewards.

This module originally lived only inside MARL_Crazyflie.ipynb's "Crazyflie
Environment" cell and depended on names defined earlier in the notebook
(imports + constants + math helpers from the "Crazyflie Config and Helpers"
cell). Those dependencies are now explicit imports below so this file can be
used standalone, outside the notebook.
"""
from ml_collections import config_dict
import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
from mujoco_playground._src import mjx_env

from constants import (
    TARGET_POS,
    QPOS_PER_DRONE,
    QVEL_PER_DRONE,
    ACTION_SIZE_PER_DRONE,
    N_PREV_ACTIONS,
    TRAINING_POS_RANGE,
    TRAINING_QUAT_RANGE,
    TRAINING_VEL_RANGE,
    TRAINING_ANG_VEL_RANGE,
    OUT_OF_BOUNDS_RANGE,
    TERMINATION_WEIGHT,
    SURVIVAL_WEIGHT,
    DRONE_PROXIMITY_WEIGHT,
    TARGET_PROXIMITY_WEIGHT,
    DRONE_PROXIMITY_THRESHOLD,
    DRONE_COLLISION_THRESHOLD,
    SCENE_PATH_2_DRONES,
)
from math_utils import clipped_normal, quat_to_rotation_matrix, add_quat_noise


# MuJoCo playground: Extending mjx_env
# https://github.com/google-deepmind/mujoco_playground/blob/main/mujoco_playground/_src/mjx_env.py
class CrazyflieEnv(mjx_env.MjxEnv):
    """MJX environment using the Crazyflie drone model."""

    def __init__(self, scene_path = SCENE_PATH_2_DRONES, num_drones = 2):
        # Define environment configuration (control and sim time steps)
        cfg = config_dict.ConfigDict()
        cfg.ctrl_dt = 0.02    # control step (50Hz)
        cfg.sim_dt = 0.002    # simulation step (500Hz)
        super().__init__(cfg)

        # Load MuJoCo model and convert to MJX
        self.scene_path = scene_path
        self._mj_model = mujoco.MjModel.from_xml_path(scene_path)
        self._mjx_model = mjx.put_model(self._mj_model)

        self.num_drones = num_drones

        # Define control/action size as number of actuators (e.g. 8 for 2 drones)
        self._action_size = self._mj_model.nu


    # Expected env properties for Brax PPO training
    @property
    def xml_path(self) -> str:
        return self.scene_path

    @property
    def action_size(self) -> int:
        return self.mjx_model.nu

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model


    def reset(self, rng: jax.Array, random_reset: bool=True, target: jax.Array=TARGET_POS) -> mjx_env.State:
        """Vectorized reset for Crazyflie drones."""

        # Random keys for state initialization and physics
        (rng,
        rng_target,
        rng_pos,
        rng_quat,
        rng_vel,
        rng_ang,
        rng_mass,
        rng_inertia,
        rng_actuator) = jax.random.split(rng, 9)


        # Random target position within TRAINING_POS_RANGE (Z >= 0.1)
        xy = jax.random.uniform(
            rng_target, (2,),
            minval=-TRAINING_POS_RANGE,
            maxval= TRAINING_POS_RANGE
        )
        z = jax.random.uniform(
            rng_target, (1,),
            minval=0.1,
            maxval=TRAINING_POS_RANGE
        )
        target_pos = jnp.concatenate([xy, z])

        # Random starting positions for drone(s) within TRAINING_POS_RANGE (Z >= 0.1)
        xy_drone = jax.random.uniform(
            rng_pos, (self.num_drones, 2),
            minval=-TRAINING_POS_RANGE,
            maxval= TRAINING_POS_RANGE
        )
        z_drone = jax.random.uniform(
            rng_pos, (self.num_drones, 1),
            minval=0.1,
            maxval=TRAINING_POS_RANGE
        )
        # Concatenate along last dimension to support batch operations
        start_pos = jnp.concatenate([xy_drone, z_drone], axis=-1)

        def enforce_min_spacing(pos, min_dist):
            """
            Ensure drones don't start too close together.

            Parameters
            ----------
            pos: Drone starting positions, shape (NUM_DRONES, 3)
            min_dist: Minimum starting distance between drones
            """
            # Pairwise differences for drone positions --> Add axis to compare drones
            # E.g. diffs[drone_1, drone_2, :] = pos[drone_1, :] - pos[drone_2, :]
            diffs = pos[:, None, :] - pos[None, :, :]      # (N, N, 3)

            # Compute magnitude of each pairwise difference, "ignore" self-difference
            # Set self-difference (e.g. drone 1 - drone 1) to large value
            dists = jnp.linalg.norm(diffs + jnp.eye(self.num_drones)[..., None] * 1e6, axis=-1)

            # Compute pair distance violation, 0 if outside of violation
            violation = jnp.maximum(0.0, min_dist - dists)  # (N, N)

            # Move each drone outside of violation along the direction of separation
            direction = diffs / (dists[..., None] + 1e-9)
            # Ensure we don't push drones downward (to avoid pushing into floor)
            direction = direction.at[..., 2].set(jnp.maximum(direction[..., 2], 0.0))
            adjust = jnp.sum(direction * violation[..., None] * 0.5, axis=1)

            return pos + adjust

        # Ensure drone spacing (don't start too close together)
        start_pos = enforce_min_spacing(start_pos, DRONE_PROXIMITY_THRESHOLD)

        # Random quaternion (different rng for axis and angle)
        rng_quat_axes, rng_quat_angles = jax.random.split(rng_quat, 2)
        std_angle = TRAINING_QUAT_RANGE / 2.0
        axes = jax.random.normal(rng_quat_axes, (self.num_drones, 3))
        axes = axes / jnp.linalg.norm(axes, axis=1, keepdims=True)
        angles = jax.random.normal(rng_quat_angles, (self.num_drones,)) * std_angle
        w = jnp.cos(angles)
        quats = jnp.concatenate([w[:, None], axes * jnp.sin(angles)[:, None]], axis=1)

        # Linear and angular velocities
        lin_vel = (TRAINING_VEL_RANGE / 2.0) * jax.random.normal(rng_vel, (self.num_drones, 3))
        ang_vel = (TRAINING_ANG_VEL_RANGE / 2.0) * jax.random.normal(rng_ang, (self.num_drones, 3))

        # Domain randomization for drone physics (clipped standard normal distribution)
        mass_noise = clipped_normal(
            rng_mass,
            self._mjx_model.body_mass.shape,
            std=1.0,
            std_clip=3.0
        )
        inertia_noise = clipped_normal(
            rng_inertia,
            self._mjx_model.body_inertia.shape,
            std=1.0,
            std_clip=3.0
        )
        actuator_noise = clipped_normal(
            rng_actuator,
            self._mjx_model.actuator_gainprm.shape,
            std=1.0,
            std_clip=3.0
        )

        # Scale noise based on property magnitude and apply it
        body_mass = self._mjx_model.body_mass * (1 + 0.05 * mass_noise)
        body_inertia = self._mjx_model.body_inertia * (1 + 0.05 * inertia_noise)
        actuator_gainprm = self._mjx_model.actuator_gainprm * (1 + 0.05 * actuator_noise)

        # Option to disable/override random stateful noise
        if not random_reset:
            # Line drones on X axis, start near floor
            x_offsets = jnp.linspace(
                -(self.num_drones - 1) / 2,
                (self.num_drones - 1) / 2,
                self.num_drones
            )
            y_offsets = jnp.zeros_like(x_offsets)
            z_offsets = jnp.full_like(x_offsets, 0.2)

            start_pos = jnp.stack([x_offsets, y_offsets, z_offsets], axis=1)

            lin_vel = jnp.zeros((self.num_drones, 3))
            quats = jnp.array([[1., 0., 0., 0.]] * self.num_drones)
            ang_vel = jnp.zeros((self.num_drones, 3))
            target_pos = target

            body_mass = self._mjx_model.body_mass
            body_inertia = self._mjx_model.body_inertia
            actuator_gainprm = self._mjx_model.actuator_gainprm

        # Assemble qpos/qvel in batch
        qpos = jnp.zeros((self.num_drones, QPOS_PER_DRONE), dtype=jnp.float32)
        qpos = qpos.at[:, 0:3].set(start_pos)
        qpos = qpos.at[:, 3:7].set(quats)
        qpos = qpos.reshape(-1)  # flatten for mjx_env

        qvel = jnp.zeros((self.num_drones, QVEL_PER_DRONE), dtype=jnp.float32)
        qvel = qvel.at[:, 0:3].set(lin_vel)
        qvel = qvel.at[:, 3:6].set(ang_vel)
        qvel = qvel.reshape(-1)  # flatten for mjx_env

        # Initialize state
        action_history = jnp.zeros((N_PREV_ACTIONS, self.num_drones, ACTION_SIZE_PER_DRONE))
        initial_distance = jnp.linalg.norm(start_pos - target_pos, axis=1)

        data = mjx_env.init(self._mjx_model, qpos=qpos, qvel=qvel)

        # Metrics
        metrics = {
            "reward/distance_improvement": jnp.zeros(()),
            "reward/proximity_bonus": jnp.zeros(()),
            "reward/survival_bonus": jnp.zeros(()),
            "reward/termination": jnp.zeros(()),
            "reward/drone_proximity_penalty": jnp.zeros(()),
            "reward": jnp.zeros(())
        }

        # State info
        info = {
            "rng": rng,
            "target_pos": target_pos,
            "initial_distance": initial_distance,
            "prev_pos": start_pos,
            "action_history": action_history, # (N_prev_actions, N_drones, Action)
            # Pass physics noise in info to step (entire mjx_model is too large)
            "body_mass": body_mass,
            "body_inertia": body_inertia,
            "actuator_gainprm": actuator_gainprm,
        }

        reward = jnp.zeros(())
        done = jnp.zeros(())

        obs = self._get_obs(data, info)

        return mjx_env.State(data=data, obs=obs, reward=reward, done=done, metrics=metrics, info=info)


    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        """Vectorized Crazyflie step for multiple drones with MARL rewards."""

        # Retrieve stateful info for drone(s)
        prev_pos = state.info["prev_pos"]
        init_dist = state.info["initial_distance"]
        target_pos = state.info["target_pos"]
        mjx_model = self._mjx_model.replace(
            body_mass = state.info["body_mass"],
            body_inertia = state.info["body_inertia"],
            actuator_gainprm = state.info["actuator_gainprm"],
        )

        # Clip flat action (by default done via XML actuators but we are also recording action history)
        action_clipped = action.at[0::4].set(jnp.clip(action[0::4], 0.0, 0.35))   # thrusts
        action_clipped = action_clipped.at[1::4].set(jnp.clip(action[1::4], -1.0, 1.0))  # rolls
        action_clipped = action_clipped.at[2::4].set(jnp.clip(action[2::4], -1.0, 1.0))  # pitches
        action_clipped = action_clipped.at[3::4].set(jnp.clip(action[3::4], -1.0, 1.0))  # yaws

        # self.n_substeps defined in mjx_env as self.dt / self.sim_dt (10 in this case)
        data = mjx_env.step(mjx_model, state.data, action_clipped, n_substeps=self.n_substeps)
        obs = self._get_obs(data, state.info)
        qpos = data.qpos.reshape(self.num_drones, QPOS_PER_DRONE)

        # Dynamic step rewards: Distance to target improvement and target proximity
        pos = qpos[:, :3]
        distance_to_target = jnp.linalg.norm(target_pos - pos, axis=1)
        distance_to_target_old = jnp.linalg.norm(target_pos - prev_pos, axis=1)
        distance_improvement = distance_to_target_old - distance_to_target

        proximity_bonus = TARGET_PROXIMITY_WEIGHT * (1 - jnp.tanh(distance_to_target))

        # Compute pairwise distances between drones
        diffs = pos[:, None, :] - pos[None, :, :]  # (N, N, 3)
        dists = jnp.linalg.norm(diffs + jnp.eye(self.num_drones)[..., None] * 1e9, axis=-1)  # (N, N)

        # Proximity penalty for being closer than threshold to another drone
        proximity_penalty_matrix = jnp.where(
            dists < DRONE_PROXIMITY_THRESHOLD,
            DRONE_PROXIMITY_WEIGHT * (1.0 - dists / DRONE_PROXIMITY_THRESHOLD),
            0.0,
        )
        # Zero out self-distances
        proximity_penalty_matrix = proximity_penalty_matrix * (1.0 - jnp.eye(self.num_drones))
        drone_proximity_penalty = jnp.sum(proximity_penalty_matrix, axis=1)

        # Collision check
        collision_matrix = (dists < DRONE_COLLISION_THRESHOLD).astype(jnp.float32)
        collision_any = jnp.clip(jnp.sum(collision_matrix * (1 - jnp.eye(self.num_drones)), axis=1), 0, 1)

        # Termination conditions: Crash, stray from target, drone collision
        done_per_drone = (
            (pos[:, 2] < 0.05)
            | (distance_to_target > init_dist + OUT_OF_BOUNDS_RANGE)
            | (collision_any > 0)
        )

        # Reward per drone, 1D jax arrays of shape (self.num_drones,)
        reward_per_drone = (
            SURVIVAL_WEIGHT
            + proximity_bonus
            + distance_improvement
            - drone_proximity_penalty
            - done_per_drone.astype(jnp.float32) * TERMINATION_WEIGHT
        )

        # Aggregate reward for the environment
        reward = jnp.sum(reward_per_drone)
        done_all = jnp.any(done_per_drone).astype(jnp.float32)

        # Aggregate per-drone metrics to scalars per environment (match Brax format)
        new_metrics = {
            "reward/distance_improvement": jnp.sum(distance_improvement),
            "reward/proximity_bonus": jnp.sum(proximity_bonus),
            "reward/survival_bonus": SURVIVAL_WEIGHT * self.num_drones,
            "reward/termination": - done_all * TERMINATION_WEIGHT,
            "reward/drone_proximity_penalty": - jnp.sum(drone_proximity_penalty),
            "reward": reward,
        }

        # Reshape flat action back to per drone, update action history
        actions_per_drone = action_clipped.reshape(self.num_drones, ACTION_SIZE_PER_DRONE)
        new_action_history = jnp.concatenate(
            [state.info["action_history"][1:],
            actions_per_drone[jnp.newaxis, ...]],
            axis=0
        )

        # Update info (prev pos and action history)
        new_info = dict(state.info)
        new_info["prev_pos"] = pos
        new_info["action_history"] = new_action_history

        return mjx_env.State(
            data=data,
            obs=obs,
            reward=reward,
            done=done_all,
            metrics=new_metrics,
            info=new_info,
        )


    def _get_obs(self, data, info):
        qpos = data.qpos.reshape(self.num_drones, QPOS_PER_DRONE)
        qvel = data.qvel.reshape(self.num_drones, QVEL_PER_DRONE)

        ##################################
        # Apply noise to base observations
        ##################################

        new_rng, pos_rng, quat_rng, vel_rng, ang_vel_rng = jax.random.split(info["rng"], 5)
        pos_noise_std = 0.01
        vel_noise_std = 0.005
        ang_vel_noise_std = 0.001
        quat_noise_std = 0.001

        # Add clipped noise (within n std) to base observations
        pos = qpos[:, :3] + clipped_normal(pos_rng, qpos[:, :3].shape, pos_noise_std, std_clip=3.0)
        quat, _ = add_quat_noise(
            quat_rng,
            qpos[:, 3:7],
            quat_noise_std,
            3.0
        )
        vel = qvel[:, :3] + clipped_normal(vel_rng, qvel[:, :3].shape, vel_noise_std, std_clip=3.0)
        ang_vel = qvel[:, 3:6] + clipped_normal(ang_vel_rng, qvel[:, 3:6].shape, ang_vel_noise_std, std_clip=3.0)

        # Update next-step rng so that it does not use the same noise
        info["rng"] = new_rng

        #########################
        # Add engineered features
        #########################

        # Convert drone(s) quat to rotation matrix
        rot_mats = jax.vmap(quat_to_rotation_matrix)(quat) # (N, 3, 3)

        # Get relative position to target in drone body coordinates (normalize by initial dist)
        initial_dist = info["initial_distance"][:, None] # (self.num_drones,) --> (N, 1)
        target_pos = info["target_pos"]
        pos_error = target_pos - pos # (N, 3)
        pos_error_norm = pos_error / (initial_dist + 1e-9)
        rel_pos_body = jnp.einsum('nij,nj->ni', rot_mats.transpose((0, 2, 1)), pos_error_norm)

        # Get linear velocity in drone body coordinates
        linear_vel_body = jnp.einsum('nij,nj->ni', rot_mats.transpose((0, 2, 1)), vel)

        # Compute relative positions to other drones in body coordinates
        # Currently includes self-relative pos (always [0, 0, 0]) - maybe remove with mask or similar
        rel_pos_all = pos[:, None, :] - pos[None, :, :]  # (N, N, 3)
        rel_pos_all_body = jnp.einsum('nij,nkj->nki', rot_mats.transpose((0, 2, 1)), rel_pos_all)  # (N, N, 3)
        rel_pos_all_body_flat = rel_pos_all_body.reshape(self.num_drones, -1)

        # Base state (per drone)
        core = jnp.concatenate(
            [
                pos, vel, ang_vel, rot_mats.reshape(self.num_drones, -1),
                rel_pos_body, linear_vel_body, rel_pos_all_body_flat
            ],
            axis=1,
        )

        # Append action history per drone
        def obs_for_drone(i):
            act_hist_flat = info["action_history"][:, i, :].reshape(-1)
            return jnp.concatenate([core[i], act_hist_flat])

        obs_per_drone = jax.vmap(obs_for_drone)(jnp.arange(self.num_drones))

        # Final flattened observation
        return obs_per_drone.reshape(-1)