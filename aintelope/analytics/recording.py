# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Repository:
# https://github.com/biological-alignment-benchmarks/biological-alignment-gridworlds-benchmarks

import base64
import csv
import logging
import os
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from filelock import FileLock

from aintelope.utils import try_df_to_csv_write

logger = logging.getLogger("aintelope.analytics.recording")

"""

"""

'''
def serialize_state(state):
    """Bool-cast, compress, base64-encode a state array for CSV storage."""
    arr = np.asarray(state, dtype=np.bool_)
    header = ",".join(str(d) for d in arr.shape).encode("ascii")
    payload = zlib.compress(arr.tobytes())
    return base64.b64encode(header + b"|" + payload).decode("ascii")


def deserialize_state(cell):
    """Reverse of serialize_state."""
    raw = base64.b64decode(cell)
    header, payload = raw.split(b"|", 1)
    shape = tuple(int(d) for d in header.decode("ascii").split(","))
    data = zlib.decompress(payload)
    return np.frombuffer(data, dtype=np.bool_).copy().reshape(shape)
'''

import pickle


def serialize_state(state):
    """Compress state for CSV storage."""
    payload = zlib.compress(pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL))
    return base64.b64encode(payload).decode("ascii")


def deserialize_state(cell):
    """Reverse of serialize_state."""
    return pickle.loads(zlib.decompress(base64.b64decode(cell)))


class EventLog(object):
    def __init__(
        self,
        experiment_dir,
        events_fname,
        headers,
    ):
        self.record_path = Path(os.path.join(experiment_dir, events_fname))
        logger.info(f"Saving training records to disk at {self.record_path}")
        self.record_path.parent.mkdir(exist_ok=True, parents=True)

        self.state_col_indices = {headers.index("State"), headers.index("Next_state")}

        write_header = not os.path.exists(self.record_path)
        self.file = open(
            self.record_path,
            mode="at",
            buffering=1024 * 1024,
            newline="",
            encoding="utf-8",
        )  # csv writer creates its own newlines therefore need to set newline to empty string here

        self.writer = csv.writer(
            self.file, quoting=csv.QUOTE_MINIMAL, delimiter=","
        )  # TODO: use TSV format instead

        if (
            write_header
        ):  # TODO: if the file already exists then assert that the header is same
            self.writer.writerow(headers)

    def log_event(self, event):
        transformed_cols = []
        for index, col in enumerate(event):
            if index in self.state_col_indices:
                col = serialize_state(col)
            elif isinstance(col, str):
                col = (
                    col.strip()
                    .replace("\r", "\\r")
                    .replace("\n", "\\n")
                    .replace("\t", "\\t")
                )  # CSV/TSV format does not support these characters

            transformed_cols.append(col)

        self.writer.writerow(transformed_cols)

    def flush(self):
        self.file.flush()

    def close(self):
        self.file.flush()
        self.file.close()


def read_events(record_path, events_filename):
    """
    Read the events saved in EventLog.
    """
    events = []

    for path in Path(record_path).rglob(events_filename):
        with FileLock(
            str(path) + ".lock"
        ):  # lock for better robustness against other processes writing to it concurrently
            df = pd.read_csv(path)
            for col in ("State", "Next_state"):
                if col in df.columns:
                    df[col] = df[col].apply(deserialize_state)
            events.append(df)

    return events


def read_checkpoints(checkpoint_dir):
    """
    Read models from a checkpoint.
    """
    model_paths = []
    for path in Path(checkpoint_dir).rglob("*"):
        model_paths.append(path)
    model_paths.sort(key=lambda x: os.path.getmtime(x))

    return model_paths
