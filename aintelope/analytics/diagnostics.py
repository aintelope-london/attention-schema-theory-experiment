# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Runtime diagnostics — used only by experiment.py during training.

Accumulates per-step learning signal and resource snapshots.
ReportCollector is the module-level singleton that owns stdout capture
and final report assembly. All structured report sections funnel through it.
"""

import io
import sys
from pathlib import Path

import pandas as pd

from aintelope.config.config_utils import TeeStream

LEARNING_COLUMNS = ["trial", "episode", "step", "loss", "epsilon", "reward"]


class ReportCollector:
    """Singleton that captures stdout and accumulates named report sections.

    Usage (orchestrator):
        collector.init(outputs_dir)          # installs TeeStream, start of run
        collector.collect({"Title": "text"}) # anywhere in the system
        collector.finalize(outputs_dir)      # writes report.txt, end of run

    Reserved key: '_elapsed' — rendered as a header line, not a titled section.
    All other keys become titled sections in insertion order.
    The raw stdout buffer is appended at the bottom of report.txt.
    """

    def __init__(self):
        self._sections = {}
        self._buf = None
        self._orig_stdout = None

    def init(self, outputs_dir):
        """Install TeeStream on sys.stdout. Call once at run start."""
        Path(outputs_dir).mkdir(parents=True, exist_ok=True)
        self._buf = io.StringIO()
        self._orig_stdout = sys.stdout
        sys.stdout = TeeStream(sys.__stdout__, self._buf)

    def collect(self, sections: dict):
        """Merge sections into the report. No-op if init() was never called."""
        self._sections.update(sections)

    def finalize(self, outputs_dir):
        """Restore stdout and write report.txt. No-op if init() was never called."""
        if self._orig_stdout is None:
            return
        sys.stdout = self._orig_stdout
        stdout_content = self._buf.getvalue()
        self._buf.close()

        parts = []
        if "_elapsed" in self._sections:
            parts.append(f"Runtime: {self._sections['_elapsed']}\n")
        for title, content in self._sections.items():
            if title.startswith("_"):
                continue
            parts.append(content + "\n")
        bar = "─" * 50
        parts.append(f"── stdout {bar[9:]}\n")
        parts.append(stdout_content)

        Path(outputs_dir).mkdir(parents=True, exist_ok=True)
        (Path(outputs_dir) / "report.txt").write_text("\n".join(parts))

        self._sections = {}
        self._buf = None
        self._orig_stdout = None


collector = ReportCollector()


class LearningMonitor:
    """Accumulates per-step loss from component update reports."""

    def __init__(self, trial: int = 0):
        self._trial = trial
        self._rows = []

    def sample(self, episode: int, step: int, report):
        """Record learning signal. Skips steps where no gradient update occurred."""
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
        """Capture resource report and send to collector as a named section.

        If collector is not initialised (write_outputs=False), falls back to
        printing directly so terminal visibility is preserved.
        """
        if collector._orig_stdout is None:
            self._resource.report()
            return
        buf = io.StringIO()
        old = sys.stdout
        sys.stdout = buf
        try:
            self._resource.report()
        finally:
            sys.stdout = old
        collector.collect({"Performance Report": buf.getvalue()})

    def performance_dataframe(self) -> pd.DataFrame:
        """Return resource snapshots as DataFrame for CSV writing by orchestrator.

        Requires ResourceMonitor.to_dataframe() — add to utils/performance.py:
            def to_dataframe(self):
                return pd.DataFrame(self._rows, columns=COLUMNS)
        """
        return self._resource.to_dataframe()

    def learning_dataframe(self) -> pd.DataFrame:
        """Return accumulated loss data as a DataFrame."""
        return self._learning.to_dataframe()
