# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Runtime diagnostics — used only by experiment.py during training.

Accumulates per-step learning signal and resource snapshots.
All analytics and reporting live in analytics/analytics.py.
"""

import pandas as pd

LEARNING_COLUMNS = ["trial", "episode", "step", "loss", "epsilon", "reward"]


class LearningMonitor:
    """Accumulates per-step loss from component update reports."""

    def __init__(self, trial: int = 0):
        self._trial = trial
        self._rows = []

    def sample(self, episode: int, step: int, report):
        """Record learning signal. Skips steps where no gradient update occurred (loss absent)."""
        loss = report.get("loss")
        if loss is not None:
            self._rows.append(
                [
                    self._trial,
                    episode,
                    step,
                    loss,
                    report.get("epsilon"),
                    report.get("reward"),
                ]
            )

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self._rows, columns=LEARNING_COLUMNS)


class DiagnosticsMonitor:
    """Coordinates resource and learning diagnostics for a single experiment block."""

    def __init__(self, context: dict):
        from aintelope.utils.performance import ResourceMonitor

        self._resource = ResourceMonitor(context)
        self._learning = LearningMonitor(trial=context.get("trial", 0))

    def sample(self, label: str):
        """Resource snapshot."""
        self._resource.sample(label)

    def sample_learning(self, episode: int, step: int, report):
        """Learning signal from an agent update report."""
        self._learning.sample(episode, step, report)

    def report(self):
        """Print resource report to terminal."""
        self._resource.report()

    def save_performance(self, folder):
        """Write performance_report.csv. Learning data is handled by analyze()."""
        self._resource.save(folder)

    def learning_dataframe(self) -> pd.DataFrame:
        """Return accumulated loss data as a DataFrame."""
        return self._learning.to_dataframe()
