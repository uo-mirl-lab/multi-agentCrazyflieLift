"""
Physical, training, and reward/observation constants shared by the GPU
(MJX/Brax) Crazyflie training pipeline.

Extracted from the "Crazyflie Config and Helpers" cell of
MARL_Crazyflie.ipynb so that crazyflie_env.py (and the other gpu_training
modules) do not depend on the notebook's global namespace.
"""
import os

import jax.numpy as jnp
import numpy as np

# ---------------------------------------------------------------------------
# Scene paths
# ---------------------------------------------------------------------------
# Default paths assume the assets_mjx folder lives alongside this file, as it
# does in this repo. In Colab, the notebook instead sets these after
# mounting Google Drive or unzipping an assets_mjx.zip upload — override
# these module attributes (or pass `scene_path=` explicitly to
# CrazyflieEnv(...)) to match that runtime.
_ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets_mjx", "bitcraze_crazyflie_2")

SCENE_PATH_1_DRONE = os.path.join(_ASSETS_DIR, "scene.xml")
SCENE_PATH_2_DRONES = os.path.join(_ASSETS_DIR, "scene_multi.xml")

# ---------------------------------------------------------------------------
# Base target position and training range
# ---------------------------------------------------------------------------
TARGET_POS = jnp.array([0.0, 0.0, 1.0])
TRAINING_POS_RANGE = 2.0

# ---------------------------------------------------------------------------
# Basic drone / action / observation sizing
# ---------------------------------------------------------------------------
BASE_HOVER_THRUST = 0.26487
QPOS_PER_DRONE = 7   # pos (3) + quat (4)
QVEL_PER_DRONE = 6   # linear vel (3) + angular vel (3)
ACTION_SIZE_PER_DRONE = 4  # thrust, roll, pitch, yaw
N_PREV_ACTIONS = 1   # number of previous actions stored in the observation

# ---------------------------------------------------------------------------
# Random starting state sampled from distribution
# ---------------------------------------------------------------------------
TRAINING_QUAT_RANGE = np.pi / 8
TRAINING_VEL_RANGE = 0.2
TRAINING_ANG_VEL_RANGE = 0.1

# Out of bounds if this much further from target compared to starting distance
OUT_OF_BOUNDS_RANGE = 2.0

# ---------------------------------------------------------------------------
# Reward and observation constants
# ---------------------------------------------------------------------------
TERMINATION_WEIGHT = 1.0
SURVIVAL_WEIGHT = TERMINATION_WEIGHT / 1000
DRONE_PROXIMITY_WEIGHT = 0.01
TARGET_PROXIMITY_WEIGHT = 0.01

DRONE_PROXIMITY_THRESHOLD = 0.2
DRONE_COLLISION_THRESHOLD = 0.05


# ---------------------------------------------------------------------------
# Model / checkpoint save locations (local training runs)
# ---------------------------------------------------------------------------
_RL_MODELS_GPU_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "rl_models", "gpu")

PPO_MODEL_SAVE_PATH = os.path.join(_RL_MODELS_GPU_DIR, "PPO_brax_model_checkpoint.pkl")
MAPPO_CHECKPOINT_DIR = os.path.join(_RL_MODELS_GPU_DIR, "MAPPO_model_checkpoint")
