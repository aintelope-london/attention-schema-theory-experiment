# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Repository:
# https://github.com/biological-alignment-benchmarks/biological-alignment-gridworlds-benchmarks

import base64
import logging
import os
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import pickle

logger = logging.getLogger("aintelope.analytics.recording")


def serialize_state(state):
    """Compress state for CSV storage."""
    payload = zlib.compress(pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL))
    return base64.b64encode(payload).decode("ascii")


def deserialize_state(cell):
    """Reverse of serialize_state."""
    return pickle.loads(zlib.decompress(base64.b64decode(cell)))


class EventLog:
    SERIALIZABLE_COLUMNS = ("State", "Next_state")

    def __init__(self, columns):
        self.columns = columns
        self._rows = []

    def log_event(self, event):
        self._rows.append(event)

    def to_dataframe(self):
        return pd.DataFrame(self._rows, columns=self.columns)

    def write(self, output_dir):
        """Write events.csv to the given directory."""
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


def read_checkpoints(checkpoint_dir):
    """
    Read models from a checkpoint.
    """
    model_paths = []
    for path in Path(checkpoint_dir).rglob("*"):
        model_paths.append(path)
    model_paths.sort(key=lambda x: os.path.getmtime(x))

    return model_paths
