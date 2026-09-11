"""
Save/load/archive helpers for Flax checkpoint directories, used to persist
and restore the MAPPO PPOTrainState.

Extracted from the "Save Model" / "Download Saved Model" / "Upload Model
Checkpoint and Import State" cells of MARL_Crazyflie.ipynb. Colab-specific
upload/download glue (google.colab.files) is left in the notebook; only the
portable save/restore/archive logic lives here.
"""
import os
import shutil

from flax.training import checkpoints


def save_checkpoint(train_state, step, ckpt_dir="./checkpoints", overwrite=True):
    """Save `train_state` to `ckpt_dir` at the given `step`. Returns the resolved absolute path."""
    ckpt_dir = os.path.abspath(ckpt_dir)
    checkpoints.save_checkpoint(ckpt_dir=ckpt_dir, target=train_state, step=step, overwrite=overwrite)
    return ckpt_dir


def archive_checkpoint(ckpt_dir="./checkpoints", archive_name="checkpoints"):
    """Zip `ckpt_dir` into `<archive_name>.zip` (e.g. for download). Returns the archive path."""
    ckpt_dir = os.path.abspath(ckpt_dir)
    return shutil.make_archive(archive_name, "zip", ckpt_dir)


def load_checkpoint(init_state, ckpt_dir):
    """Restore a checkpoint directory into a freshly-initialized `init_state` pytree (same structure)."""
    ckpt_dir = os.path.abspath(ckpt_dir)
    return checkpoints.restore_checkpoint(ckpt_dir=ckpt_dir, target=init_state)
