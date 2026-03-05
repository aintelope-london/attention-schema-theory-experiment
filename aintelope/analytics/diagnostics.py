# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Diagnostics coordination — learning signal monitoring and run reporting."""

from pathlib import Path
import pandas as pd

from aintelope.analytics.recording import write_csv
from aintelope.utils.performance import ResourceMonitor

LEARNING_COLUMNS = ["episode", "step", "loss"]


# ── Pure analytics ────────────────────────────────────────────────────


def compute_learning_analytics(
    events: pd.DataFrame,
    episode_fraction: float = 0.15,
    min_improvement_ratio: float = 1.3,
) -> dict:
    """Compute learning improvement metrics from an EventLog DataFrame.

    Returns a dict keyed by phase ('train', 'test'), each containing:
    ratio, start_avg, end_avg, window, passed, min_improvement_ratio.
    """
    result = {}
    for phase, mask in [("train", ~events["IsTest"]), ("test", events["IsTest"])]:
        phase_events = events[mask]
        if phase_events.empty:
            continue
        episode_rewards = phase_events.groupby("Episode")["Reward"].sum().sort_index()
        n = len(episode_rewards)
        window = max(1, int(n * episode_fraction))
        start_avg = episode_rewards.iloc[:window].mean()
        end_avg = episode_rewards.iloc[-window:].mean()
        if start_avg <= 0:
            ratio = None
            passed = end_avg > start_avg
        else:
            ratio = end_avg / start_avg
            passed = ratio >= min_improvement_ratio
        result[phase] = {
            "ratio": ratio,
            "start_avg": float(start_avg),
            "end_avg": float(end_avg),
            "window": window,
            "passed": passed,
            "min_improvement_ratio": min_improvement_ratio,
        }
    return result


def sample_episodes(
    events: pd.DataFrame,
    metric_col: str = "Reward",
    every_n: int = 10,
    agg: str = "sum",
) -> pd.Series:
    """Sample per-episode metric aggregates at regular intervals.

    Generic over metric column, aggregation function, and sampling frequency.
    Returns a Series indexed by episode number.
    """
    episode_totals = (
        events.groupby("Episode")[metric_col]
        .agg(agg)
        .sort_index()
    )
    return episode_totals.iloc[::every_n]


def _format_architecture(architecture: dict) -> str:
    """Render architecture dict as a compact one-liner."""
    parts = []
    for cid, entry in architecture.items():
        inputs = ",".join(entry.get("inputs", []))
        parts.append(f"{cid}={entry['type']}({inputs})")
    return " | ".join(parts)


def format_run_report(analytics: dict, context: dict) -> str:
    """Format a full run report string from analytics and run context."""
    lines = ["── Run Summary ──────────────────────────────────────"]

    # Identity
    lines.append(f"outputs_dir: {context.get('outputs_dir', '?')}")
    lines.append(
        f"trials: {context.get('trials', '?')}"
        f" | episodes: {context.get('episodes', '?')}"
        f" | steps: {context.get('steps', '?')}"
    )
    lines.append("")

    # Agent and architecture
    lines.append(f"Agent:    {context.get('agent_class', '?')}")
    if "architecture" in context:
        lines.append(f"Arch:     {_format_architecture(context['architecture'])}")

    # Env
    env_parts = []
    for k in ("map_max", "combine_interoception_and_vision", "env_layout_seed_repeat_sequence_length"):
        if k in context:
            env_parts.append(f"{k}={context[k]}")
    if env_parts:
        lines.append(f"Env:      {' | '.join(env_parts)}")

    # Training
    train_parts = []
    for k in ("gamma", "batch_size", "lr"):
        if k in context:
            train_parts.append(f"{k}={context[k]}")
    if train_parts:
        lines.append(f"Training: {' | '.join(train_parts)}")

    # Reward samples
    if "reward_samples" in context:
        samples = context["reward_samples"]
        every_n = context.get("reward_sample_every_n", "?")
        sample_str = "  ".join(f"ep{ep}:{val:.1f}" for ep, val in samples.items())
        lines.append(f"\nReward samples (every {every_n} eps):")
        lines.append(f"  {sample_str}")

    # Learning improvement
    lines.append("\n── Learning Improvement ────────────────────────────")
    for phase, metrics in analytics.items():
        status = "✓" if metrics["passed"] else "✗"
        if metrics["ratio"] is not None:
            lines.append(
                f"  {phase.capitalize()}: {status} ratio={metrics['ratio']:.2f}x"
                f" (start={metrics['start_avg']:.3f}, end={metrics['end_avg']:.3f},"
                f" window={metrics['window']} episodes)"
            )
        else:
            lines.append(
                f"  {phase.capitalize()}: {status}"
                f" start={metrics['start_avg']:.3f} → end={metrics['end_avg']:.3f}"
            )

    lines.append("────────────────────────────────────────────────────")
    return "\n".join(lines)


def write_run_report(
    analytics: dict, events: pd.DataFrame, context: dict, folder: Path
) -> None:
    """Print and write report.txt + learning_curve.png to the run's root folder."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    report = format_run_report(analytics, context)
    print(report)
    (folder / "report.txt").write_text(report)
    _write_learning_curve(events, folder)


def _write_learning_curve(events: pd.DataFrame, folder: Path) -> None:
    """Plot per-episode reward curves per phase and write learning_curve.png."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    for phase, mask, color in [
        ("train", ~events["IsTest"], "steelblue"),
        ("test", events["IsTest"], "darkorange"),
    ]:
        phase_events = events[mask]
        if phase_events.empty:
            continue
        episode_rewards = phase_events.groupby("Episode")["Reward"].sum().sort_index()
        ax.plot(episode_rewards.index, episode_rewards.values, alpha=0.25, color=color)
        window = max(1, len(episode_rewards) // 10)
        rolling = episode_rewards.rolling(window).mean()
        ax.plot(
            rolling.index, rolling.values,
            color=color, label=phase.capitalize(), linewidth=2,
        )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("Learning Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(folder / "learning_curve.png", dpi=100)
    plt.close(fig)


# ── Learning signal accumulator ───────────────────────────────────────


class LearningMonitor:
    """Accumulates per-step learning signal from component update reports."""

    def __init__(self):
        self._rows = []

    def sample(self, episode, step, report):
        """Record a learning signal. Silently skips if report is None or has no loss."""
        if report is None:
            return
        loss = report.get("loss") if isinstance(report, dict) else None
        if loss is not None:
            self._rows.append([episode, step, loss])

    def to_dataframe(self):
        return pd.DataFrame(self._rows, columns=LEARNING_COLUMNS)


# ── Coordinator ───────────────────────────────────────────────────────


class DiagnosticsMonitor:
    """Coordinates resource and learning diagnostics for a single experiment block."""

    def __init__(self, context):
        self._resource = ResourceMonitor(context)
        self._learning = LearningMonitor()

    def sample(self, label):
        """Resource snapshot — same interface as ResourceMonitor."""
        self._resource.sample(label)

    def sample_learning(self, episode, step, report):
        """Learning signal from an agent update report."""
        self._learning.sample(episode, step, report)

    def report(self):
        """Print resource report to terminal."""
        self._resource.report()

    def save(self, folder):
        """Write per-block diagnostics: performance_report.csv + learning.csv."""
        folder = Path(folder)
        self._resource.save(folder)
        df = self._learning.to_dataframe()
        if not df.empty:
            write_csv(folder / "learning.csv", df)