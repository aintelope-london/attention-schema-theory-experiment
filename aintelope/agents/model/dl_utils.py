"""Checkpoint path construction and resolution for model persistence."""

import glob
from pathlib import Path


def checkpoint_path(outputs_dir, agent_id, i_trial):
    """Return the canonical save path for an agent's checkpoint.

    Format: {outputs_dir}/checkpoints/{agent_id}_trial_{i_trial}.pt
    """
    return Path(outputs_dir) / "checkpoints" / f"{agent_id}_trial_{i_trial}.pt"


def select_checkpoint(outputs_dir, agent_id, i_trial, custom_model=""):
    """Resolve which checkpoint to load for a given agent and trial.

    Resolution order:
        1. custom_model path if set → use it directly
        2. Glob checkpoints/{agent_id}_*.pt → sort alphabetically → index by i_trial % n
        3. None → fresh initialization
    """
    if custom_model:
        path = Path(custom_model)
        return path if path.exists() else None

    trial_ckpt = checkpoint_path(outputs_dir, agent_id, i_trial)
    if trial_ckpt.exists():
        return trial_ckpt
    return None
