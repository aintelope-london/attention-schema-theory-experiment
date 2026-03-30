# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Event recording and run discovery for experiment outputs."""

import ast
import base64
import os
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import pickle

# Columns stored as compressed+base64 blobs — require deserialize_state on read.
SERIALIZABLE_COLUMNS = ("Observation", "Board")

# Columns stored as Python repr strings (e.g. "(1, 2)") — require ast.literal_eval on read.
TUPLE_COLUMNS = ("Position", "Food_position")

STATE_COLUMNS = ["Run_id", "Trial", "Episode", "Step", "Board"]


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
        df = pd.DataFrame(self._rows, columns=self.columns)
        for col in SERIALIZABLE_COLUMNS:
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda x: serialize_state(x) if x is not None else None
                )
        return df


class StateLog:
    """Per-step environment state accumulator. One row per step, not per agent."""

    def __init__(self):
        self.columns = STATE_COLUMNS
        self._rows = []

    def log(self, row):
        self._rows.append(row)

    def to_dataframe(self):
        df = pd.DataFrame(self._rows, columns=self.columns)
        for col in SERIALIZABLE_COLUMNS:
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda x: serialize_state(x) if x is not None else None
                )
        return df


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
        write_csv(Path(outputs_dir) / name / filename, df)


def write_results(outputs_dir, events, states):
    """Write event and state DataFrames grouped by experiment to disk."""
    _write_grouped_csv(outputs_dir, events, "events.csv")
    _write_grouped_csv(outputs_dir, states, "states.csv")


def read_events(filepath):
    """Read an events CSV back into a DataFrame with all columns fully parsed.

    - SERIALIZABLE_COLUMNS are decompressed from base64 blobs.
    - TUPLE_COLUMNS are parsed from their string repr (e.g. "(1, 2)").
    All consumers receive uniform Python objects regardless of source path.
    """
    df = pd.read_csv(filepath)
    for col in SERIALIZABLE_COLUMNS:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: deserialize_state(x) if pd.notna(x) else None
            )
    for col in TUPLE_COLUMNS:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: ast.literal_eval(x) if pd.notna(x) else None
            )
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


def save_env_layout(image, outputs_dir, seed):
    path = Path(outputs_dir) / "env_layouts" / f"{seed}.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def save_frames(frames, output_path, frame_duration=0.7):
    """Save PIL Image frames to a video or animated image file.

    Format is inferred from the output_path extension (.mp4, .gif).

    Args:
        frames: List of PIL.Image.Image.
        output_path: Output file path.
        frame_duration: Seconds each frame is displayed.
    """
    ext = Path(output_path).suffix.lower()
    if ext == ".gif":
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=int(frame_duration * 1000),
            loop=0,
        )
    else:
        import imageio

        fps = 10
        repeats = round(frame_duration * fps)
        images = [np.array(frame) for frame in frames for _ in range(repeats)]
        imageio.mimwrite(output_path, images, fps=fps)