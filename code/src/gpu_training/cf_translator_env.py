"""
CrazyflieTranslatorEnv: a hierarchical "sim-to-real translator" environment.

Wraps a low-level PPO hover policy with a high-level physics "translator"
policy that outputs a per-channel scale [thrust_scale, roll_scale,
pitch_scale, yaw_scale] applied to the low-level policy's raw action, 
with the idea being that this policy will respond to physics differences
between sim and real (but is trained on sim2sim).

Observation (flat), for history_len == N:
    pos_history    : (N, 3) past drone positions
    rotmat_history : (N, 3, 3) past drone rotation matrices
    action_history : (N, 4) past raw low-level actions (pre-scale)
Aligned so action_history[i] is the action that produced pos_history[i] /
rotmat_history[i] -- action i caused state i+1. The action about to be
chosen this step is never included.

Implements the same mjx_env.MjxEnv-style interface as CrazyflieEnv, so it's
a drop-in `environment=` for ppo_training.py and rollout_utils.rollout_policy.

Known limitation: always call rollout_policy(...) on this env with
print_logs=False -- its per-step printing assumes CrazyflieEnv's own obs
layout, not this env's history-based one (wrong numbers, not a crash).
"""
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
import mujoco
from mujoco_playground._src import mjx_env
from ml_collections import config_dict

from constants import ACTION_SIZE_PER_DRONE, SCENE_PATH_1_DRONE, TARGET_POS
from crazyflie_env import CrazyflieEnv
from math_utils import clipped_normal, quat_to_rotation_matrix


@dataclass
class TranslatorDomainRandomization:
    """Per-episode "sim-to-real gap" ranges the translator has to correct
    for. Defaults are much larger than CrazyflieEnv.reset()'s own (+/-5%)
    noise -- the point is to be harder/more varied than what the low-level
    policy trained under."""

    # Multiplicative mismatch on the low-level policy's raw action, on top of
    # the translator's own scale -- e.g. sim hover thrust 0.3 vs. real 0.2 is
    # a ~0.67 gap; a translator_scale of ~1/gap would compensate for it.
    thrust_gap_range: Tuple[float, float] = (0.5, 1.8)
    torque_gap_range: Tuple[float, float] = (0.6, 1.6)

    # Extra mass/inertia/actuator-gain noise, layered on top of (replacing)
    # CrazyflieEnv's own smaller reset() noise (see reset() below).
    mass_noise_std: float = 0.20
    inertia_noise_std: float = 0.20
    actuator_noise_std: float = 0.20

    # Constant per-episode wind-like force (N) / torque (N*m) at the drone
    # body's center of mass -- a steady breeze, not gusts.
    wind_force_std: float = 0.03
    wind_torque_std: float = 0.0005

    # Max std (radians) of the gravity direction tilt (magnitude preserved) --
    # e.g. a propeller/motor not pushing quite where it should.
    gravity_tilt_std: float = 0.05

    @classmethod
    def nominal(cls):
        """No extra randomization beyond CrazyflieEnv's own baseline reset()
        noise -- gap gains pinned to 1.0 (no mismatch), everything else 0.
        Useful as a sanity check that a wrapped policy still flies under the
        same conditions the low-level policy was trained under."""
        return cls(
            thrust_gap_range=(1.0, 1.0),
            torque_gap_range=(1.0, 1.0),
            mass_noise_std=0.0,
            inertia_noise_std=0.0,
            actuator_noise_std=0.0,
            wind_force_std=0.0,
            wind_torque_std=0.0,
            gravity_tilt_std=0.0,
        )


def _tilt_gravity(rng, gravity, tilt_std):
    """Rotate `gravity` by a small random angle about a random axis perpendicular to it (Rodrigues' formula)."""
    axis_rng, angle_rng = jax.random.split(rng)

    axis = jax.random.normal(axis_rng, (3,))
    # Project out the component parallel to gravity so the axis is a "horizontal" tilt axis.
    axis = axis - jnp.dot(axis, gravity) / (jnp.dot(gravity, gravity) + 1e-9) * gravity
    axis = axis / (jnp.linalg.norm(axis) + 1e-9)

    angle = clipped_normal(angle_rng, (), std=tilt_std, std_clip=3.0)

    g_rot = (
        gravity * jnp.cos(angle)
        + jnp.cross(axis, gravity) * jnp.sin(angle)
        + axis * jnp.dot(axis, gravity) * (1 - jnp.cos(angle))
    )
    return g_rot


def _remap_scale(action, scale_min, scale_max):
    """Map a raw policy output (Brax's tanh_normal actions are bounded to
    (-1, 1)) to [scale_min, scale_max], with 0 mapping to 1.0 (no
    correction). A plain clip(action, scale_min, scale_max) would collapse
    every negative action to scale_min and make anything above 1.0
    unreachable -- this keeps the full range reachable and keeps an
    untrained (near-zero-mean) policy starting near "no correction" instead
    of near "zero everything out"."""
    pos = 1.0 + action * (scale_max - 1.0)
    neg = 1.0 + action * (1.0 - scale_min)
    return jnp.clip(jnp.where(action >= 0, pos, neg), scale_min, scale_max)


class CrazyflieTranslatorEnv(mjx_env.MjxEnv):
    """High-level "translator" environment wrapping a frozen low-level hover policy."""

    def __init__(
        self,
        low_level_inference_fn: Callable,
        scene_path: str = SCENE_PATH_1_DRONE,
        history_len: int = 4,
        scale_range: Tuple[float, float] = (0.5, 2.0),
        randomization: Optional[TranslatorDomainRandomization] = None,
        randomize_gravity: bool = True,
        drone_body_name: str = "cf2",
    ):
        # low_level_inference_fn: frozen, jit-ready Brax-style policy
        # `(obs, rng) -> (action, extras)`, e.g.
        # make_inference_fn(params, deterministic=True) from an
        # already-trained single-drone PPO run. Not updated during training.
        cfg = config_dict.ConfigDict()
        cfg.ctrl_dt = 0.02   # control step (50Hz), matches CrazyflieEnv
        cfg.sim_dt = 0.002   # simulation step (500Hz), matches CrazyflieEnv
        super().__init__(cfg)

        self._low_level_env = CrazyflieEnv(scene_path=scene_path, num_drones=1)
        self._low_level_inference_fn = low_level_inference_fn

        self.history_len = history_len
        self.scale_min, self.scale_max = scale_range
        self.randomization = randomization or TranslatorDomainRandomization()
        self.randomize_gravity = randomize_gravity

        self._drone_body_id = mujoco.mj_name2id(
            self._low_level_env.mj_model, mujoco.mjtObj.mjOBJ_BODY, drone_body_name.encode()
        )
        if self._drone_body_id < 0:
            raise ValueError(
                f"Body '{drone_body_name}' not found in {scene_path}; "
                "pass drone_body_name= explicitly for this scene."
            )

        # Translator observation: history_len * (pos(3) + rotmat(9) + action(4))
        self._obs_size = self.history_len * (3 + 9 + ACTION_SIZE_PER_DRONE)

    # -- Expected env properties for Brax PPO training / rollout_utils -----

    @property
    def xml_path(self) -> str:
        return self._low_level_env.xml_path

    @property
    def action_size(self) -> int:
        # [thrust_scale, roll_scale, pitch_scale, yaw_scale]
        return ACTION_SIZE_PER_DRONE

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._low_level_env.mj_model

    @property
    def mjx_model(self):
        return self._low_level_env.mjx_model

    @property
    def num_drones(self) -> int:
        # The translator wraps exactly one low-level drone.
        return 1

    # -- Core env API --------------------------------------------------

    def _get_translator_obs(self, pos_history, rotmat_history, action_history):
        return jnp.concatenate([
            pos_history.reshape(-1),
            rotmat_history.reshape(-1),
            action_history.reshape(-1),
        ])

    def reset(self, rng: jax.Array, random_reset: bool = True, target: jax.Array = TARGET_POS) -> mjx_env.State:
        """Reset the wrapped low-level env and sample this episode's sim-to-real gap."""
        (rng,
         low_rng,
         thrust_gap_rng,
         torque_gap_rng,
         mass_rng,
         inertia_rng,
         actuator_rng,
         wind_force_rng,
         wind_torque_rng,
         gravity_rng) = jax.random.split(rng, 10)

        low_state = self._low_level_env.reset(low_rng, random_reset=random_reset, target=target)
        low_info = dict(low_state.info)
        dr = self.randomization
        base_model = self._low_level_env.mjx_model

        # --- Per-episode action-space gap (the star of the show) ---
        thrust_gap = jax.random.uniform(
            thrust_gap_rng, (), minval=dr.thrust_gap_range[0], maxval=dr.thrust_gap_range[1]
        )
        torque_gap = jax.random.uniform(
            torque_gap_rng, (3,), minval=dr.torque_gap_range[0], maxval=dr.torque_gap_range[1]
        )
        low_info["action_gap_gain"] = jnp.concatenate([thrust_gap.reshape(1), torque_gap])

        # --- Extra (larger) mass/inertia/actuator-gain domain randomization,
        # replacing the small-noise values CrazyflieEnv.reset() already sampled ---
        mass_noise = clipped_normal(mass_rng, base_model.body_mass.shape, std=1.0, std_clip=3.0)
        inertia_noise = clipped_normal(inertia_rng, base_model.body_inertia.shape, std=1.0, std_clip=3.0)
        actuator_noise = clipped_normal(actuator_rng, base_model.actuator_gainprm.shape, std=1.0, std_clip=3.0)

        # Layer on top of (not replace) the noise CrazyflieEnv.reset() already
        # sampled -- with std=0.0 (TranslatorDomainRandomization.nominal())
        # this is a true no-op, leaving CrazyflieEnv's own noise untouched.
        low_info["body_mass"] = low_info["body_mass"] * (1 + dr.mass_noise_std * mass_noise)
        low_info["body_inertia"] = low_info["body_inertia"] * (1 + dr.inertia_noise_std * inertia_noise)
        low_info["actuator_gainprm"] = low_info["actuator_gainprm"] * (1 + dr.actuator_noise_std * actuator_noise)

        # --- Constant per-episode wind-like force/torque at the drone body ---
        wind_force = clipped_normal(wind_force_rng, (3,), std=dr.wind_force_std, std_clip=3.0)
        wind_torque = clipped_normal(wind_torque_rng, (3,), std=dr.wind_torque_std, std_clip=3.0)
        xfrc = jnp.zeros_like(low_state.data.xfrc_applied)
        xfrc = xfrc.at[self._drone_body_id, 0:3].set(wind_force)
        xfrc = xfrc.at[self._drone_body_id, 3:6].set(wind_torque)
        data = low_state.data.replace(xfrc_applied=xfrc)

        # --- Optional small gravity tilt ---
        if self.randomize_gravity:
            low_info["gravity"] = _tilt_gravity(gravity_rng, base_model.opt.gravity, dr.gravity_tilt_std)

        pos = data.qpos[0:3]
        rotmat = quat_to_rotation_matrix(data.qpos[3:7])

        action_history = jnp.zeros((self.history_len, ACTION_SIZE_PER_DRONE))
        pos_history = jnp.tile(pos, (self.history_len, 1))
        rotmat_history = jnp.tile(rotmat.reshape(1, -1), (self.history_len, 1))

        translator_obs = self._get_translator_obs(pos_history, rotmat_history, action_history)

        info = {
            "rng": rng,
            "low_level_info": low_info,
            "low_level_metrics": low_state.metrics,
            # Cache the low-level obs so step() can query the low-level
            # policy without recomputing it (as elsewhere in this codebase).
            "low_level_obs": low_state.obs,
            "action_history": action_history,
            "pos_history": pos_history,
            "rotmat_history": rotmat_history,
            # Mirrored up so state.info has the same "target_pos" key
            # CrazyflieEnv's does -- rollout_policy() reads it to render
            # the target marker.
            "target_pos": low_info["target_pos"],
        }

        metrics = dict(low_state.metrics)
        # Same keys step() produces -- Brax's PPO scan carry needs identical
        # metrics structure from reset() through every step().
        metrics["translator/mean_scale"] = jnp.zeros(())
        metrics["translator/thrust_scale"] = jnp.zeros(())

        return mjx_env.State(
            data=data,
            obs=translator_obs,
            reward=jnp.zeros(()),
            done=jnp.zeros(()),
            metrics=metrics,
            info=info,
        )

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        """
        Query the frozen low-level policy, scale its raw action by the
        translator's action (and the hidden episode gap gain), step the
        wrapped low-level env, and update the translator's own history obs.
        """
        scale = _remap_scale(action, self.scale_min, self.scale_max)

        low_info = state.info["low_level_info"]
        low_metrics = state.info["low_level_metrics"]
        low_level_obs = state.info["low_level_obs"]

        rng, act_rng = jax.random.split(state.info["rng"])
        raw_action, _ = self._low_level_inference_fn(low_level_obs, act_rng)

        # Real world applies raw_action * scale * gap, but the translator
        # never sees the gap directly -- only its effect on the history.
        action_gap_gain = low_info["action_gap_gain"]
        effective_action = raw_action * scale * action_gap_gain

        low_state = mjx_env.State(
            data=state.data,
            obs=low_level_obs,
            reward=jnp.zeros(()),
            done=jnp.zeros(()),
            metrics=low_metrics,
            info=low_info,
        )
        next_low_state = self._low_level_env.step(low_state, effective_action)

        pos = next_low_state.data.qpos[0:3]
        rotmat = quat_to_rotation_matrix(next_low_state.data.qpos[3:7])

        # action_history[i] produced pos_history[i]/rotmat_history[i] --
        # action i caused state i+1; the next action is never included here.
        action_history = jnp.concatenate([state.info["action_history"][1:], raw_action[None, :]], axis=0)
        pos_history = jnp.concatenate([state.info["pos_history"][1:], pos[None, :]], axis=0)
        rotmat_history = jnp.concatenate([state.info["rotmat_history"][1:], rotmat.reshape(1, -1)], axis=0)

        translator_obs = self._get_translator_obs(pos_history, rotmat_history, action_history)

        new_info = dict(state.info)
        new_info["rng"] = rng
        new_info["low_level_info"] = next_low_state.info
        new_info["low_level_metrics"] = next_low_state.metrics
        new_info["low_level_obs"] = next_low_state.obs
        new_info["action_history"] = action_history
        new_info["pos_history"] = pos_history
        new_info["rotmat_history"] = rotmat_history
        new_info["target_pos"] = next_low_state.info["target_pos"]

        # Reuse the low-level env's own reward/termination -- translator is
        # rewarded for the same task despite the injected sim-to-real gap.
        metrics = dict(next_low_state.metrics)
        metrics["translator/mean_scale"] = jnp.mean(scale)
        metrics["translator/thrust_scale"] = scale[0]

        return mjx_env.State(
            data=next_low_state.data,
            obs=translator_obs,
            reward=next_low_state.reward,
            done=next_low_state.done,
            metrics=metrics,
            info=new_info,
        )
