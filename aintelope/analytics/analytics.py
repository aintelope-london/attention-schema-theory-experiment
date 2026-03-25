# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Analytics — entry point for all run diagnostics.

analyze(results) orchestrates metric computation, figure rendering, and file
output. Returns {metric_name: {block: data_dict}} for test assertions.

Figures are written to disk when cfg.run.write_outputs is True.
The GUI renders its own figures interactively from loaded CSVs via PLOT_TYPES.
"""

import numpy as np
from pathlib import Path

from aintelope.analytics.diagnostics import collector
from aintelope.analytics.metrics import (
    _METRICS,
    _episode_windows,
    per_episode_efficiency,
    text,
)
from aintelope.analytics.plot_primitives import (
    create_figure,
    create_figure_grid,
    get_color,
    plot_series,
    render_bar,
    render_heatmap,
    render_scatter,
    save_figure,
)
from aintelope.analytics.recording import write_csv


# ── Output helpers ────────────────────────────────────────────────────────────


def _cfg(results):
    return next(iter(results.values()))["cfg"]


def _write_figure(cfg, name, fig):
    if cfg.run.write_outputs:
        path = Path(cfg.run.outputs_dir) / f"{name}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        save_figure(fig, path)


def _write_df(cfg, name, df):
    if cfg.run.write_outputs and not df.empty:
        write_csv(Path(cfg.run.outputs_dir) / f"{name}.csv", df)


# ── Series rendering ──────────────────────────────────────────────────────────

_SERIES_SPECS = {
    "learning_curve": ("Episode", "Total Reward", "Learning Curve"),
    "loss_curve": ("Episode", "Mean Loss", "Loss Curve"),
    "epsilon_curve": ("Episode", "Epsilon", "Epsilon Decay"),
    "reward_curve": ("Episode", "Mean Reward", "Reward Signal"),
    "steps_to_reward": ("Episode", "Steps to First Reward", "Steps to Reward"),
    "efficiency_curve": ("Episode", "Efficiency", "Policy Efficiency over Training"),
    "roi_food_alignment": (
        "Episode",
        "Food in ROI rate",
        "ROI Food Alignment over Training",
    ),
}


def _render_series(metric_data, x_label, y_label, title):
    series = {
        block: data["series"]
        for block, data in metric_data.items()
        if data.get("series") is not None
    }
    if not series:
        return None
    return plot_series(series, x_label, y_label, title)


# ── Per-block rendering ───────────────────────────────────────────────────────


def _render_visitation_heatmap(block_data):
    valid = block_data["events"]
    n_windows = block_data["n_windows"]
    block = block_data["block"]
    if valid.empty:
        return None
    rows_vals = valid["Position"].apply(lambda p: p[0])
    cols_vals = valid["Position"].apply(lambda p: p[1])
    grid_h = int(rows_vals.max()) + 1
    grid_w = int(cols_vals.max()) + 1
    windows = _episode_windows(valid["Episode"].unique(), n_windows)
    figure, axes = create_figure_grid(len(windows))
    for ax, (label, ep_set) in zip(axes, windows):
        mask = valid["Episode"].isin(ep_set)
        grid = np.zeros((grid_h, grid_w))
        for r, c in zip(rows_vals[mask], cols_vals[mask]):
            grid[int(r), int(c)] += 1
        render_heatmap(ax, grid, f"{block} — {label}")
    figure.tight_layout()
    return figure


def _render_action_distribution(block_data):
    valid = block_data["events"]
    action_names = block_data["action_names"]
    n_windows = block_data["n_windows"]
    block = block_data["block"]
    if valid.empty:
        return None
    all_actions = sorted(valid["Action"].unique())
    labels = [action_names.get(a, str(a)) for a in all_actions]
    windows = _episode_windows(valid["Episode"].unique(), n_windows)
    figure, axes = create_figure_grid(len(windows))
    for i, (ax, (label, ep_set)) in enumerate(zip(axes, windows)):
        counts = (
            valid[valid["Episode"].isin(ep_set)]["Action"]
            .value_counts()
            .reindex(all_actions, fill_value=0)
        )
        total = counts.sum()
        fractions = (counts / total).values if total > 0 else counts.values
        render_bar(ax, labels, fractions, f"{block} — {label}", get_color(i))
    figure.tight_layout()
    return figure


def _render_roi_turn_distribution(block_data):
    valid = block_data["events"]
    n_windows = block_data["n_windows"]
    block = block_data["block"]
    if valid.empty:
        return None
    all_actions = sorted(valid["Internal_action"].unique())
    labels = [str(int(a)) for a in all_actions]
    windows = _episode_windows(valid["Episode"].unique(), n_windows)
    figure, axes = create_figure_grid(len(windows))
    for i, (ax, (label, ep_set)) in enumerate(zip(axes, windows)):
        counts = (
            valid[valid["Episode"].isin(ep_set)]["Internal_action"]
            .value_counts()
            .reindex(all_actions, fill_value=0)
        )
        total = counts.sum()
        fractions = (counts / total).values if total > 0 else counts.values
        render_bar(ax, labels, fractions, f"{block} — {label}", get_color(i))
    figure.tight_layout()
    return figure


def _render_optimality_scatter(block_data):
    per_episode = block_data["per_episode"]
    reached = [e for e in per_episode if e["steps_to_goal"] != float("inf")]
    if not reached:
        return None
    figure, ax = create_figure()
    xs = [e["spawn_dist"] for e in reached]
    ys = [e["steps_to_goal"] for e in reached]
    render_scatter(ax, xs, ys, "Optimality Scatter")
    return figure


# Maps metric name → renderer called with the full block_data dict → figure, one per block.
_BLOCK_RENDERERS = {
    "visitation_heatmap": _render_visitation_heatmap,
    "action_distribution": _render_action_distribution,
    "roi_turn_distribution": _render_roi_turn_distribution,
    "optimal_efficiency": _render_optimality_scatter,
}


# ── Entry point ───────────────────────────────────────────────────────────────


def analyze(results):
    """Run all configured analytics. Returns {name: {block: data_dict}}.

    Figures are written to disk as a side-effect when cfg.run.write_outputs is True.
    """
    cfg = _cfg(results)
    out = {}

    for name, params in cfg.run.analytics.items():
        metric_data = _METRICS[name](results, params)
        out[name] = metric_data

        if name in _SERIES_SPECS:
            x_label, y_label, title = _SERIES_SPECS[name]
            fig = _render_series(metric_data, x_label, y_label, title)
            if fig:
                _write_figure(cfg, name, fig)

        elif name in _BLOCK_RENDERERS:
            render_fn = _BLOCK_RENDERERS[name]
            for block, block_data in metric_data.items():
                fig = render_fn(block_data)
                if fig:
                    _write_figure(cfg, f"{name}_{block}", fig)

    return out


# ── Assertion helpers ─────────────────────────────────────────────────────────


def assert_learning_improvement(block_result):
    """Assert a single block's learning improvement passed."""
    if not block_result["passed"]:
        if block_result["ratio"] is not None:
            raise AssertionError(
                f"Insufficient improvement: ratio={block_result['ratio']:.2f}x"
                f" < {block_result['min_improvement_ratio']}x"
                f" (start={block_result['start_avg']:.3f},"
                f" end={block_result['end_avg']:.3f},"
                f" window={block_result['window']} episodes)"
            )
        else:
            raise AssertionError(
                f"No improvement: start_avg={block_result['start_avg']:.3f},"
                f" end_avg={block_result['end_avg']:.3f}"
            )


def report_optimal_policy(block_result):
    """Print per-episode optimality table and assert mean efficiency >= threshold.

    Printing here is for pytest terminal visibility only — the structured report
    is already in report.txt via the optimal_efficiency metric.
    """
    print("\n── Optimal Policy Report ─────────────────────────────")
    for ep in block_result["per_episode"]:
        dist = ep["spawn_dist"] if ep["spawn_dist"] is not None else "?"
        steps = ep["steps_to_goal"] if ep["steps_to_goal"] is not None else "never"
        eff = (
            f"{ep['efficiency'] * 100:.0f}%" if ep["efficiency"] is not None else "N/A"
        )
        print(
            f"  Episode {ep['episode']:>4}: spawn_dist={dist}, steps_to_goal={steps}, efficiency={eff}"
        )
    mean_eff = block_result["efficiency_pct"]
    n = block_result["n_episodes"]
    suffix = f"(mean over {n} episodes)"
    print(
        f"\n  Efficiency: {mean_eff:.1f}% {suffix}"
        if mean_eff is not None
        else f"\n  Efficiency: N/A {suffix}"
    )
    print("──────────────────────────────────────────────────────\n")
    assert (
        mean_eff is not None and mean_eff >= block_result["min_efficiency_pct"]
    ), f"Policy efficiency {mean_eff:.1f}% < required {block_result['min_efficiency_pct']:.1f}%"
    return mean_eff
