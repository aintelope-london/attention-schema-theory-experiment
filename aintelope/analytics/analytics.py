# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Analytics — single entry point for all learning diagnostics and run reporting.

Architecture:
  analyze(cfg, events, learning_df) → AnalyticsResult
    ↓ iterates _ANALYTICS registry
      each analytic: prepare (inline) → plot() → text() → contribution dict
    ↓ write_analytics(result, folder) writes figures / report.txt / CSVs

Shared render helpers (plot, text) are the only bridge between analytics
logic and plotting.py. Analytics never import matplotlib directly.

Analytic library is appended at the bottom of this file — one function per
analytic, registered with @register_analytic.
"""

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from aintelope.analytics.plotting import (
    aggregate_series,
    collapse,
    create_figure,
    get_color,
    plot_band,
    save_figure,
)
from aintelope.analytics.recording import write_csv

# ── Result container ──────────────────────────────────────────────────────────


@dataclass
class AnalyticsResult:
    """All outputs from analyze(). Consumed by orchestrator and tests."""

    metrics: dict = field(default_factory=dict)
    figures: dict = field(default_factory=dict)  # name → Figure
    texts: dict = field(default_factory=dict)  # name → str  (assembled → report.txt)
    dataframes: dict = field(default_factory=dict)  # name → DataFrame


# ── Registry ──────────────────────────────────────────────────────────────────

_ANALYTICS = []  # list of (name, fn, config_flag)


def register_analytic(name, flag=None):
    """Decorator. flag: cfg.run.analytics attribute name, or None = always run."""

    def decorator(fn):
        _ANALYTICS.append((name, fn, flag))
        return fn

    return decorator


# ── Shared render helpers ─────────────────────────────────────────────────────


def plot(series_by_label, x_label, y_label, title, ref_line=None):
    """Render a multi-series mean±std band plot. Returns Figure.

    series_by_label: {label: (x_arr, mean_arr, std_arr)}
    ref_line: optional (label, y_value) for a horizontal reference line.
    """
    figure, ax = create_figure()
    for i, (label, (x, mean, std)) in enumerate(series_by_label.items()):
        plot_band(ax, x, mean, std, label=label, color=get_color(i))
    if ref_line is not None:
        ref_label, ref_y = ref_line
        ax.axhline(ref_y, linestyle="--", color="grey", linewidth=1.2, label=ref_label)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    if series_by_label or ref_line:
        ax.legend()
    ax.figure.tight_layout()
    return figure


def text(title, rows):
    """Render a labelled text block for report.txt.

    rows: list of str lines.
    """
    bar = "─" * 50
    header = f"── {title} " + bar[len(title) + 4 :]
    return "\n".join([header] + rows)


# ── Core compute functions ────────────────────────────────────────────────────


def compute_learning_analytics(
    events: pd.DataFrame,
    episode_fraction: float = 0.15,
    min_improvement_ratio: float = 1.3,
) -> dict:
    """Pure. Returns {phase: {ratio, start_avg, end_avg, window, passed, ...}}."""
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


def assert_learning_improvement(analytics: dict, phase: str = "train") -> None:
    """Assert learning improvement from pre-computed analytics dict."""
    metrics = analytics.get(phase)
    assert metrics is not None, f"No '{phase}' phase data in analytics"
    if not metrics["passed"]:
        if metrics["ratio"] is not None:
            raise AssertionError(
                f"Insufficient improvement: ratio={metrics['ratio']:.2f}x"
                f" < {metrics['min_improvement_ratio']}x"
                f" (start={metrics['start_avg']:.3f}, end={metrics['end_avg']:.3f},"
                f" window={metrics['window']} episodes)"
            )
        else:
            raise AssertionError(
                f"No improvement: start_avg={metrics['start_avg']:.3f},"
                f" end_avg={metrics['end_avg']:.3f}"
            )


def compute_steps_to_first_reward(events: pd.DataFrame) -> pd.DataFrame:
    """Per-episode step index of first positive reward.

    Returns DataFrame[Episode, Trial, steps_to_reward].
    """
    positive = events[events["Reward"] > 0][["Episode", "Trial", "Step"]]
    if positive.empty:
        return pd.DataFrame(columns=["Episode", "Trial", "steps_to_reward"])
    first = positive.groupby(["Episode", "Trial"])["Step"].min().reset_index()
    first.columns = ["Episode", "Trial", "steps_to_reward"]
    return first


def compute_optimal_manhattan(events: pd.DataFrame):
    """Per-episode Manhattan distance from agent spawn to food spawn.

    Reads Position and Food_position at Step 0 (pre-action spawn coordinates).
    Returns Series indexed by (Episode, Trial), or None if columns absent.
    """
    if "Position" not in events.columns or "Food_position" not in events.columns:
        return None
    step0 = events[events["Step"] == 0][
        ["Episode", "Trial", "Position", "Food_position"]
    ].dropna()
    if step0.empty:
        return None

    def manhattan(row):
        try:
            ar, ac = row["Position"]
            fr, fc = row["Food_position"]
            return abs(ar - fr) + abs(ac - fc)
        except Exception:
            return None

    step0 = step0.copy()
    step0["optimal_steps"] = step0.apply(manhattan, axis=1)
    return step0.set_index(["Episode", "Trial"])["optimal_steps"].dropna()


def sample_episodes(
    events: pd.DataFrame,
    metric_col: str = "Reward",
    every_n: int = 10,
    agg: str = "sum",
) -> pd.Series:
    """Sample per-episode metric aggregates at regular intervals."""
    episode_totals = events.groupby("Episode")[metric_col].agg(agg).sort_index()
    return episode_totals.iloc[::every_n]


# ── Entry point ───────────────────────────────────────────────────────────────


def analyze(
    cfg, events: pd.DataFrame, learning_df: pd.DataFrame = None
) -> AnalyticsResult:
    """Run all configured analytics. Returns AnalyticsResult.

    Always computes metrics (improvement ratio).
    Per-analytic flags in cfg.run.analytics gate optional outputs.
    """
    result = AnalyticsResult()
    result.metrics = compute_learning_analytics(
        events,
        episode_fraction=cfg.run.analytics.episode_fraction,
        min_improvement_ratio=cfg.run.analytics.min_improvement_ratio,
    )
    analytics_cfg = cfg.run.analytics
    for name, fn, flag in _ANALYTICS:
        if flag is None or getattr(analytics_cfg, flag, False):
            contribution = fn(events, learning_df, cfg)
            result.figures.update(contribution.get("figures", {}))
            result.texts.update(contribution.get("texts", {}))
            result.dataframes.update(contribution.get("dataframes", {}))
    return result


def write_analytics(result: AnalyticsResult, folder) -> None:
    """Write AnalyticsResult to disk: figures (.png), report.txt, dataframes (.csv)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for name, fig in result.figures.items():
        save_figure(fig, folder / f"{name}.png")
    if result.texts:
        report = "\n\n".join(v for _, v in sorted(result.texts.items()))
        (folder / "report.txt").write_text(report)
    for name, df in result.dataframes.items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            write_csv(folder / f"{name}.csv", df)


# ── Helpers shared by library functions ──────────────────────────────────────


def _format_architecture(architecture: dict) -> str:
    parts = []
    for cid, entry in architecture.items():
        inputs = ",".join(entry.get("inputs", []))
        parts.append(f"{cid}={entry['type']}({inputs})")
    return " | ".join(parts)


# ── Analytic library ──────────────────────────────────────────────────────────
# Convention: each function signature is (events, learning_df, cfg) → contribution dict
# Keys in contribution: "figures", "texts", "dataframes" (all optional).
# Use plot() and text() helpers for output; never import matplotlib here.
# Prefix text keys with sort order ("00_", "01_", ...) to control report order.


@register_analytic("run_summary")
def analytic_run_summary(events, learning_df, cfg):
    """Textual run metadata block."""
    agent_cfg = cfg.agent_params.agent_0
    lines = [f"Agent:    {agent_cfg.agent_class}"]
    if hasattr(agent_cfg, "architecture"):
        arch = {
            cid: {"type": entry.type, "inputs": list(entry.inputs)}
            for cid, entry in agent_cfg.architecture.items()
        }
        lines.append(f"Arch:     {_format_architecture(arch)}")
    env_keys = (
        "map_max",
        "combine_interoception_and_vision",
        "env_layout_seed_repeat_sequence_length",
    )
    env_parts = [
        f"{k}={getattr(cfg.env_params, k)}"
        for k in env_keys
        if hasattr(cfg.env_params, k)
    ]
    if env_parts:
        lines.append(f"Env:      {' | '.join(env_parts)}")
    train_parts = []
    for attr, label in [
        ("gamma", "γ"),
        ("batch_size", "batch"),
        ("learning_rate", "lr"),
    ]:
        if hasattr(cfg.agent_params, attr):
            train_parts.append(f"{label}={getattr(cfg.agent_params, attr)}")
    if train_parts:
        lines.append(f"Training: {' | '.join(train_parts)}")
    return {"texts": {"00_run_summary": text("Run Summary", lines)}}


@register_analytic("learning_improvement")
def analytic_learning_improvement(events, learning_df, cfg):
    """Learning improvement ratio text block."""
    metrics = compute_learning_analytics(
        events,
        episode_fraction=cfg.run.analytics.episode_fraction,
        min_improvement_ratio=cfg.run.analytics.min_improvement_ratio,
    )
    lines = []
    for phase, m in metrics.items():
        status = "✓" if m["passed"] else "✗"
        if m["ratio"] is not None:
            lines.append(
                f"  {phase.capitalize()}: {status}  ratio={m['ratio']:.2f}x"
                f"  (start={m['start_avg']:.3f}, end={m['end_avg']:.3f},"
                f" window={m['window']} eps)"
            )
        else:
            lines.append(
                f"  {phase.capitalize()}: {status}"
                f"  start={m['start_avg']:.3f} → end={m['end_avg']:.3f}"
            )
    return {"texts": {"01_learning_improvement": text("Learning Improvement", lines)}}


@register_analytic("learning_curve")
def analytic_learning_curve(events, learning_df, cfg):
    """Episode × reward band plot, one series per phase."""
    series = {}
    for phase, mask in [("Train", ~events["IsTest"]), ("Test", events["IsTest"])]:
        phase_df = events[mask]
        if phase_df.empty:
            continue
        collapsed = collapse(phase_df, ["Episode", "Trial"], "Reward", "sum")
        series[phase] = aggregate_series(collapsed, "Episode", "Reward")
    if not series:
        return {}
    fig = plot(
        series, x_label="Episode", y_label="Total Reward", title="Learning Curve"
    )
    return {"figures": {"learning_curve": fig}}


@register_analytic("loss_curve", flag="loss_curve")
def analytic_loss_curve(events, learning_df, cfg):
    """Episode × mean loss band plot."""
    if learning_df is None or learning_df.empty:
        return {}
    df = learning_df.copy()
    if "trial" not in df.columns:
        df["trial"] = 0
    collapsed = collapse(df, ["episode", "trial"], "loss", "mean")
    collapsed.columns = ["Episode", "Trial", "Loss"]
    series = {"Loss": aggregate_series(collapsed, "Episode", "Loss")}
    fig = plot(series, x_label="Episode", y_label="Mean Loss", title="Loss Curve")
    return {"figures": {"loss_curve": fig}, "dataframes": {"learning": learning_df}}


@register_analytic("steps_to_reward", flag="steps_to_reward")
def analytic_steps_to_reward(events, learning_df, cfg):
    """Steps to first reward per episode, vs optional optimal Manhattan baseline."""
    train_events = events[~events["IsTest"]]
    steps_df = compute_steps_to_first_reward(train_events)
    if steps_df.empty:
        return {}

    series = {
        "Steps to reward": aggregate_series(steps_df, "Episode", "steps_to_reward")
    }
    ref_line = None
    figures = {}
    texts_out = {}
    dfs = {"steps_to_reward": steps_df}

    if getattr(cfg.run.analytics, "optimal_policy_pct", False):
        optimal = compute_optimal_manhattan(train_events)
        if optimal is not None:
            med_opt = float(optimal.median())
            ref_line = ("Optimal (Manhattan)", med_opt)

            steps_idx = steps_df.set_index(["Episode", "Trial"])["steps_to_reward"]
            pct = (med_opt / steps_idx * 100).clip(0, 100).reset_index()
            pct.columns = ["Episode", "Trial", "pct_optimal"]
            figures["pct_optimal"] = plot(
                {"% of optimal": aggregate_series(pct, "Episode", "pct_optimal")},
                x_label="Episode",
                y_label="% of Optimal Policy",
                title="Optimality (%)",
            )
            dfs["pct_optimal"] = pct

            med_actual = float(steps_df["steps_to_reward"].median())
            pct_val = med_opt / med_actual * 100 if med_actual > 0 else 0.0
            texts_out["02_steps_to_reward"] = text(
                "Steps to Reward",
                [
                    f"  Median steps (actual):  {med_actual:.1f}",
                    f"  Optimal (Manhattan):    {med_opt:.1f}",
                    f"  Efficiency:             {pct_val:.1f}%",
                ],
            )

    figures["steps_to_reward"] = plot(
        series,
        x_label="Episode",
        y_label="Steps to First Reward",
        title="Steps to Reward",
        ref_line=ref_line,
    )
    return {"figures": figures, "texts": texts_out, "dataframes": dfs}
