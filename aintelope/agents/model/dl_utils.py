"""Checkpoint path construction and resolution for model persistence."""

import glob
from pathlib import Path


def checkpoint_path(outputs_dir, agent_id, i_trial):
    """Return the canonical save path for an agent's checkpoint.

    Format: {outputs_dir}/checkpoints/{agent_id}_trial_{i_trial}.pt
    Creates the directory if needed.
    """
    path = Path(outputs_dir) / "checkpoints" / f"{agent_id}_trial_{i_trial}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


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

    pattern = str(Path(outputs_dir) / "checkpoints" / f"{agent_id}_*.pt")
    found = sorted(glob.glob(pattern))
    if found:
        return Path(found[i_trial % len(found)])
    return None
