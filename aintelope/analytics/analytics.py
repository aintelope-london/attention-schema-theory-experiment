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

import numpy as np
import pandas as pd

from aintelope.analytics.plotting import (
    aggregate_series,
    collapse,
    create_figure,
    get_color,
    plot_band,
    save_figure,
)
from aintelope.analytics.recording import (
    write_csv,
    SERIALIZABLE_COLUMNS,
    deserialize_state,
)

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


def register_analytic(name):
    """Decorator. flag: cfg.run.analytics attribute name, or None = always run."""

    def decorator(fn):
        _ANALYTICS.append((name, fn))
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


def _dist_to_nearest(vision, channel) -> int | None:
    """Min Manhattan distance from viewport center to nearest target in given channel.

    Args:
        vision: (C, H, W) numpy array.
        channel: index of the target channel (e.g. food_ind from manifesto).

    Returns:
        Integer Manhattan distance, or None if no target visible.
    """
    h, w = vision.shape[1], vision.shape[2]
    cr, cc = h // 2, w // 2
    coords = list(zip(*np.where(vision[channel] > 0)))
    if not coords:
        return None
    return min(abs(r - cr) + abs(c - cc) for r, c in coords)


def compute_learning_analytics(
    events: pd.DataFrame,
    episode_fraction: float = 0.15,
    min_improvement_ratio: float = 1.3,
) -> dict:
    """Pure. Returns {phase: {ratio, start_avg, end_avg, window, passed, ...}}."""
    result = {}
    for phase in ["train", "test"]:
        phase_events = filter_events(events, IsTest=(phase == "test"))
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


def filter_events(events: pd.DataFrame, **filters) -> pd.DataFrame:
    # Design: events travel compressed throughout (SERIALIZABLE_COLUMNS contain
    # pickled+zlib bytes). Deserialization is deferred to here so the pipe and
    # in-memory DataFrame stay small. All analytics that need observation data
    # must access it through this function — never index events directly.
    # Callers are responsible for keeping filters selective enough to fit in memory.
    mask = pd.Series(True, index=events.index)
    for col, val in filters.items():
        mask &= events[col] == val
    result = events[mask].copy()
    for col in SERIALIZABLE_COLUMNS:
        if col in result.columns:
            result[col] = result[col].apply(
                lambda x: deserialize_state(x) if x is not None else None
            )
    return result


def compute_optimal_analytics(
    events: pd.DataFrame,
    target_channel: int,
    phase: str = "test",
) -> dict:
    """Per-episode efficiency of reaching the nearest target for the first time.

    Efficiency is defined as spawn_dist / steps_to_first_reward, where:
      - spawn_dist: Manhattan distance from agent to nearest target at step 0
        (read from the vision channel; approximate spawn distance).
      - steps_to_first_reward: step index of first positive reward in episode.

    efficiency = 1.0 means the agent took the minimum possible steps.
    efficiency = 0.0 means the agent never reached the target.

    The target_channel argument is agnostic — any goal visible in the vision
    array can be evaluated by passing the appropriate channel index.

    Args:
        events: Combined events DataFrame from run().
        target_channel: Vision channel index for the goal (e.g. manifesto["food_ind"]).
        phase: "test" or "train".

    Returns:
        dict with efficiency_pct (mean %), per_episode list, n_episodes.
    """
    phase_events = filter_events(events, IsTest=(phase == "test"))
    if phase_events.empty:
        return {"efficiency_pct": None, "per_episode": [], "n_episodes": 0}

    steps_by_episode = (
        compute_steps_to_first_reward(phase_events)
        .groupby("Episode")["steps_to_reward"]
        .min()
    )

    per_episode = []
    for episode, ep_events in phase_events.groupby("Episode"):
        step0_rows = ep_events[ep_events["Step"] == 0]
        spawn_dist = None
        if not step0_rows.empty:
            obs = step0_rows.iloc[0]["Observation"]
            if obs is not None:
                spawn_dist = _dist_to_nearest(obs["vision"], target_channel)

        steps_to_goal = (
            int(steps_by_episode[episode])
            if episode in steps_by_episode.index
            else None
        )

        if spawn_dist is None:
            efficiency = None
        elif steps_to_goal is None:
            efficiency = 0.0
        elif steps_to_goal == 0 or spawn_dist == 0:
            efficiency = 1.0
        else:
            efficiency = min(1.0, spawn_dist / steps_to_goal)

        per_episode.append(
            {
                "episode": int(episode),
                "spawn_dist": spawn_dist,
                "steps_to_goal": steps_to_goal,
                "efficiency": efficiency,
            }
        )

    valid = [e["efficiency"] for e in per_episode if e["efficiency"] is not None]
    mean_eff = float(np.mean(valid)) * 100 if valid else 0.0

    return {
        "efficiency_pct": mean_eff,
        "per_episode": per_episode,
        "n_episodes": len(per_episode),
    }


def report_optimal_policy(optimal: dict) -> float:
    """Print per-episode optimality table and assert mean efficiency >= threshold.

    Always prints — designed for test output visibility.

    Args:
        optimal: dict returned by compute_optimal_analytics().
        min_efficiency_pct: Minimum acceptable mean efficiency (0–100).

    Returns:
        Mean efficiency %.

    Raises:
        AssertionError if mean efficiency < min_efficiency_pct.
    """
    print("\n── Optimal Policy Report ─────────────────────────────")
    for ep in optimal["per_episode"]:
        dist = ep["spawn_dist"] if ep["spawn_dist"] is not None else "?"
        steps = ep["steps_to_goal"] if ep["steps_to_goal"] is not None else "never"
        eff = (
            f"{ep['efficiency'] * 100:.0f}%" if ep["efficiency"] is not None else "N/A"
        )
        print(
            f"  Episode {ep['episode']:>4}: spawn_dist={dist}, steps_to_goal={steps}, efficiency={eff}"
        )

    mean_eff = optimal["efficiency_pct"]
    n = optimal["n_episodes"]
    suffix = f"(mean over {n} episodes)"
    if mean_eff is None:
        print(f"\n  Efficiency: N/A {suffix}")
    else:
        print(f"\n  Efficiency: {mean_eff:.1f}% {suffix}")
    print("──────────────────────────────────────────────────────\n")
    min_efficiency_pct = optimal["min_efficiency_pct"]
    assert (
        mean_eff is not None and mean_eff >= min_efficiency_pct
    ), f"Policy efficiency {mean_eff:.1f}% < required {min_efficiency_pct:.1f}%"
    return mean_eff


def _compute_optimal_manhattan(events: pd.DataFrame):
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
    cfg,
    events: pd.DataFrame,
    learning_df: pd.DataFrame = None,
    manifesto: dict = None,
) -> AnalyticsResult:
    """Run all configured analytics. Returns AnalyticsResult.

    Always computes learning improvement metrics.
    Computes optimal policy metrics when manifesto provides a target channel.
    Per-analytic flags in cfg.run.analytics gate optional plot/text outputs.
    """
    result = AnalyticsResult()
    result.metrics = compute_learning_analytics(
        events,
        episode_fraction=cfg.run.analytics.episode_fraction,
        min_improvement_ratio=cfg.run.analytics.min_improvement_ratio,
    )

    if manifesto is not None and manifesto.get("food_ind") is not None:
        result.metrics["optimal"] = compute_optimal_analytics(
            events,
            manifesto["food_ind"],
            min_efficiency_pct=cfg.run.analytics.min_efficiency_pct,
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
    for phase in ["Train", "Test"]:
        phase_df = filter_events(events, IsTest=(phase.lower() == "test"))
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


@register_analytic("epsilon_curve")
def analytic_epsilon_curve(events, learning_df, cfg):
    """Epsilon decay over training episodes."""
    if learning_df is None or learning_df.empty or "epsilon" not in learning_df.columns:
        return {}
    df = learning_df.copy()
    collapsed = collapse(df, ["episode", "trial"], "epsilon", "mean")
    collapsed.columns = ["Episode", "Trial", "Epsilon"]
    series = {"Epsilon": aggregate_series(collapsed, "Episode", "Epsilon")}
    fig = plot(series, x_label="Episode", y_label="Epsilon", title="Epsilon Decay")
    return {"figures": {"epsilon_curve": fig}}


@register_analytic("reward_curve")
def analytic_reward_curve(events, learning_df, cfg):
    """Per-update reward signal over training episodes."""
    if learning_df is None or learning_df.empty or "reward" not in learning_df.columns:
        return {}
    df = learning_df.copy()
    collapsed = collapse(df, ["episode", "trial"], "reward", "mean")
    collapsed.columns = ["Episode", "Trial", "Reward"]
    series = {"Reward": aggregate_series(collapsed, "Episode", "Reward")}
    fig = plot(series, x_label="Episode", y_label="Mean Reward", title="Reward Signal")
    return {"figures": {"reward_curve": fig}}


@register_analytic("loss_curve")
def analytic_loss_curve(events, learning_df, cfg):
    """Episode x mean loss band plot."""
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


@register_analytic("steps_to_reward")
def analytic_steps_to_reward(events, learning_df, cfg):
    """Steps to first reward per episode, vs optional optimal Manhattan baseline."""
    train_events = filter_events(events, IsTest=False)
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
        optimal = _compute_optimal_manhattan(train_events)
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
