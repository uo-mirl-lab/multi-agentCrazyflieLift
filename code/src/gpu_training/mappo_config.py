"""Hyperparameter configuration for the custom Multi-Agent PPO (MAPPO) trainer
in mappo_trainer.py. Bundled into a dataclass so multiple configs (e.g. for
sweeps) can coexist."""
from dataclasses import dataclass
from typing import Tuple


@dataclass
class MAPPOConfig:
    """Hyperparameters for MAPPO training. Defaults match the notebook's original values."""

    num_drones: int = 2

    # Rollout / batching
    num_envs: int = 2048
    num_updates: int = 1200
    rollout_steps: int = 128
    num_epochs: int = 2
    num_minibatches: int = 16
    episode_length: int = 1500

    # Discounting
    gamma: float = 0.995
    gae_lambda: float = 0.95

    # Optimization (linearly annealed LR)
    lr: float = 3e-3
    lr_final_frac: float = 0.25  # lr_final = lr * lr_final_frac

    clip_eps: float = 0.2
    entropy_coef: float = 1e-3
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    seed: int = 42
    policy_hidden: Tuple[int, ...] = (128, 128)
    # Larger hidden layers for the critic since it sees all drones' observations
    vf_hidden: Tuple[int, ...] = (256, 256)

    @property
    def lr_final(self) -> float:
        return self.lr * self.lr_final_frac

    @property
    def mini_batch_size(self) -> int:
        return self.num_envs * self.rollout_steps // self.num_minibatches
