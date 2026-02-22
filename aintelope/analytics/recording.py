"""Event recording and run discovery for experiment outputs."""

import base64
import os
import zlib
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import pickle

SERIALIZABLE_COLUMNS = ("State", "Next_state", "Observation", "Board")
STATE_COLUMNS = ["Run_id", "Trial", "Episode", "Step", "Board"]


def get_checkpoint(outputs_dir: str, agent_id: str) -> Optional[Path]:
    """Return existing checkpoint path for an agent, or None."""
    path = Path(outputs_dir) / "checkpoints" / f"{agent_id}.pt"
    return path if path.exists() else None


def checkpoint_path(outputs_dir: str, agent_id: str) -> Path:
    """Return the checkpoint save path for an agent, ensuring the directory exists."""
    path = Path(outputs_dir) / "checkpoints" / f"{agent_id}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def serialize_state(state):
    """Compress state for CSV storage."""
    payload = zlib.compress(pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL))
    return base64.b64encode(payload).decode("ascii")


def deserialize_state(cell):
    """Reverse of serialize_state."""
    return pickle.loads(zlib.decompress(base64.b64decode(cell)))


class EventLog:
    """In-experiment accumulator. Does not escape experiments.py."""

    def __init__(self, columns):
        self.columns = columns
        self._rows = []

    def log_event(self, event):
        self._rows.append(event)

    def to_dataframe(self):
        return pd.DataFrame(self._rows, columns=self.columns)


class StateLog:
    """Per-step environment state accumulator. One row per step, not per agent."""

    def __init__(self):
        self._rows = []

    def log(self, row):
        self._rows.append(row)

    def to_dataframe(self):
        return pd.DataFrame(self._rows, columns=STATE_COLUMNS)


def write_csv(path, df):
    """Write a DataFrame to CSV. Creates parent directories."""
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)
    df.to_csv(path, index=False)


def _write_grouped_csv(outputs_dir, frames, filename):
    """Write DataFrames grouped by Run_id to per-block CSV files."""
    combined = pd.concat(frames, ignore_index=True)
    for name, group in combined.groupby("Run_id"):
        df = group.copy()
        for col in SERIALIZABLE_COLUMNS:
            if col in df.columns:
                df[col] = df[col].apply(serialize_state)
        write_csv(Path(outputs_dir) / name / filename, df)


def write_results(outputs_dir, events, states):
    """Write event and state DataFrames grouped by experiment to disk."""
    _write_grouped_csv(outputs_dir, events, "events.csv")
    _write_grouped_csv(outputs_dir, states, "states.csv")


def read_events(filepath):
    """Read an events CSV back into a DataFrame, deserializing state columns."""
    df = pd.read_csv(filepath)
    for col in SERIALIZABLE_COLUMNS:
        if col in df.columns:
            df[col] = df[col].apply(deserialize_state)
    return df


def list_runs(outputs_dir):
    """Return run directory names under outputs_dir, newest first."""
    outputs_path = Path(outputs_dir)
    return sorted(
        [d.name for d in outputs_path.iterdir() if d.is_dir()],
        reverse=True,
    )


def list_blocks(run_dir):
    """Return block names within a run that contain events.csv."""
    run_path = Path(run_dir)
    return sorted(d.name for d in run_path.iterdir() if (d / "events.csv").exists())


def read_checkpoints(checkpoint_dir):
    """Read models from a checkpoint."""
    model_paths = sorted(
        Path(checkpoint_dir).rglob("*"),
        key=lambda x: os.path.getmtime(x),
    )
    return model_paths


def frames_to_video(frames, output_path, frame_duration=0.7):
    """Render PIL Image frames as an mp4 video.

    Args:
        frames: List of PIL.Image.Image.
        output_path: Output .mp4 file path.
        frame_duration: Seconds each frame is displayed.
    """
    import imageio

    images = [np.array(frame) for frame in frames]
    imageio.mimwrite(output_path, images, fps=1.0 / frame_duration)
