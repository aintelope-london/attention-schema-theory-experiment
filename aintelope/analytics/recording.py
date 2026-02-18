"""Event recording and run discovery for experiment outputs."""

import base64
import json
import os
import zlib
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import pickle
from collections import defaultdict


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
    SERIALIZABLE_COLUMNS = ("State", "Next_state", "Observation")

    def __init__(self, columns):
        self.columns = columns
        self._rows = []
        self.experiment_name = None

    def log_event(self, event):
        self._rows.append(event)

    def to_dataframe(self):
        return pd.DataFrame(self._rows, columns=self.columns)

    def write(self, output_dir):
        """Write events.csv and meta.json to the given directory."""
        path = Path(output_dir) / "events.csv"
        path.parent.mkdir(exist_ok=True, parents=True)
        df = self.to_dataframe()
        for col in self.SERIALIZABLE_COLUMNS:
            if col in df.columns:
                df[col] = df[col].apply(serialize_state)
        df.to_csv(path, index=False)


    @staticmethod
    def read(filepath):
        df = pd.read_csv(filepath)
        for col in EventLog.SERIALIZABLE_COLUMNS:
            if col in df.columns:
                df[col] = df[col].apply(deserialize_state)
        return df


def write_results(outputs_dir, event_logs):
    """Merge EventLogs by experiment and write each to disk."""
    by_experiment = defaultdict(list)
    for log in event_logs:
        by_experiment[log.experiment_name].append(log)
    for name, logs in by_experiment.items():
        combined = EventLog(logs[0].columns)
        for log in logs:
            combined._rows.extend(log._rows)
        combined.write(str(Path(outputs_dir) / name))


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
    """
    Read models from a checkpoint.
    """
    model_paths = []
    for path in Path(checkpoint_dir).rglob("*"):
        model_paths.append(path)
    model_paths.sort(key=lambda x: os.path.getmtime(x))

    return model_paths


def frames_to_video(frames, output_path, frame_duration=0.7, font_size=20):
    """Render text frames as an mp4 video.

    Args:
        frames: List of frames, each a list of strings (one per grid row).
        output_path: Output .mp4 file path.
        frame_duration: Seconds each frame is displayed.
        font_size: Font size for text rendering.
    """
    from PIL import Image, ImageDraw, ImageFont
    import imageio

    font = ImageFont.load_default(size=font_size)

    # Measure cell size from widest/tallest character across all frames
    measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    chars = {ch for lines in frames for line in lines for ch in line}
    cell_w = max(measure.textbbox((0, 0), ch, font=font)[2] for ch in chars)
    cell_h = max(measure.textbbox((0, 0), ch, font=font)[3] for ch in chars)

    rows = len(frames[0])
    cols = max(len(line) for lines in frames for line in lines)
    padding = 10
    size = (cols * cell_w + 2 * padding, rows * cell_h + 2 * padding)

    images = []
    for lines in frames:
        img = Image.new("RGB", size, color="black")
        draw = ImageDraw.Draw(img)
        for y, line in enumerate(lines):
            for x, ch in enumerate(line):
                draw.text(
                    (padding + x * cell_w, padding + y * cell_h),
                    ch,
                    fill="white",
                    font=font,
                )
        images.append(np.array(img))

    imageio.mimwrite(output_path, images, fps=1.0 / frame_duration)
