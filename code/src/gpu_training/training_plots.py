"""Stateful progress-plotting helpers for the PPO (Brax) and MAPPO training
loops. Both call a `progress_fn(step, metrics)` callback periodically; these
classes hold the accumulated plot state so multiple training runs don't
stomp on each other's state."""
from datetime import datetime

import matplotlib.pyplot as plt
from IPython.display import clear_output, display


def linear_annealing_lr(init_lr, final_lr, num_timesteps):
    """Linearly anneal a learning rate from `init_lr` to `final_lr` over `num_timesteps` steps."""
    def schedule(step):
        progress = step / num_timesteps
        return init_lr + progress * (final_lr - init_lr)
    return schedule


class PPOProgressPlotter:
    """
    Tracks and plots Brax PPO eval-reward curves across training. Pass an
    instance as `progress_fn` to `brax.training.agents.ppo.train`.
    """

    def __init__(self, num_timesteps, y_lim=(-20, 20)):
        self.num_timesteps = num_timesteps
        self.y_lim = y_lim
        self.x_data = []
        self.y_data = []
        self.y_dataerr = []
        self.times = [datetime.now()]

    def __call__(self, num_steps, metrics):
        clear_output(wait=True)
        self.times.append(datetime.now())
        self.x_data.append(num_steps)
        self.y_data.append(metrics["eval/episode_reward"])
        self.y_dataerr.append(metrics["eval/episode_reward_std"])

        plt.xlim([0, self.num_timesteps * 1.25])
        plt.ylim(list(self.y_lim))
        plt.xlabel("# environment steps")
        plt.ylabel("reward per episode")
        plt.title(f"y={self.y_data[-1]:.3f}")
        plt.errorbar(self.x_data, self.y_data, yerr=self.y_dataerr, color="blue")
        display(plt.gcf())

    def print_timing(self):
        """Print jit vs. total training wall-clock time (requires >=2 calls)."""
        print(f"time to jit: {self.times[1] - self.times[0]}")
        print(f"time to train: {self.times[-1] - self.times[1]}")


class MAPPOProgressPlotter:
    """
    Tracks and plots mean-reward curves for the custom MAPPO trainer. Pass
    an instance as `progress_fn` to `mappo_trainer.train(...)`.
    """

    def __init__(self, num_updates, y_lim=(-25, 25)):
        self.num_updates = num_updates
        self.y_lim = y_lim
        self.x_data = []
        self.y_data = []

    def __call__(self, update, metrics):
        clear_output(wait=True)

        self.x_data.append(update)
        self.y_data.append(metrics["mean_reward"])

        plt.clf()
        plt.plot(self.x_data, self.y_data, marker='o', linewidth=2)

        plt.xlim([0, self.num_updates * 1.25])
        plt.ylim(list(self.y_lim))
        plt.xlabel("# updates")
        plt.ylabel("mean reward")
        plt.title(f"mean_reward = {self.y_data[-1]:.3f}")

        plt.grid(True)
        display(plt.gcf())
