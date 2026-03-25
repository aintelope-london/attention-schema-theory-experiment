# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Registered plot types for the results viewer GUI.

Each function renders into a provided axes object and is registered via
@register_plot, populating PLOT_TYPES for the GUI dropdown.

Signature: (ax, df, metric, groups, group_col)
  ax        — matplotlib Axes to render into
  df        — events DataFrame for the selected block
  metric    — column name selected in GUI (ignored by plots that compute their own data)
  groups    — list of group values (e.g. agent IDs) selected in GUI
  group_col — column name for grouping (e.g. "Agent_id")
"""

import numpy as np

from aintelope.analytics.metrics import (
    first_reward,
    per_episode_efficiency,
    _episode_windows,
)
from aintelope.analytics.plot_primitives import (
    aggregate_series,
    collapse,
    get_color,
    plot_grouped_bands,
    register_plot,
    render_bar,
    render_heatmap,
    render_scatter,
)


@register_plot("Learning Curve")
def plot_learning_curve(ax, df, metric, groups, group_col):
    """Episode × metric, summed per episode, mean±std across trials."""

    def series(group_df, m):
        collapsed = collapse(group_df, ["Episode", "Trial"], m, "sum")
        return aggregate_series(collapsed, "Episode", m)

    plot_grouped_bands(ax, df, metric, groups, group_col, series, "Episode")


@register_plot("Step Profile")
def plot_step_profile(ax, df, metric, groups, group_col):
    """Step × metric, mean±std across episodes and trials."""

    def series(group_df, m):
        return aggregate_series(group_df, "Step", m)

    plot_grouped_bands(ax, df, metric, groups, group_col, series, "Step")


@register_plot("Trial Spread")
def plot_trial_spread(ax, df, metric, groups, group_col):
    """Trial × metric, summed per episode, mean±std of episodes per trial."""

    def series(group_df, m):
        collapsed = collapse(group_df, ["Trial", "Episode"], m, "sum")
        return aggregate_series(collapsed, "Trial", m)

    plot_grouped_bands(ax, df, metric, groups, group_col, series, "Trial")


@register_plot("Visitation Heatmap")
def plot_visitation_heatmap(ax, df, metric, groups, group_col):
    """Position visit count heatmap across all episodes in the selected block."""
    ax.clear()
    valid = df[df["Position"].apply(lambda x: x is not None)]
    if valid.empty:
        return
    rows_vals = valid["Position"].apply(lambda p: p[0])
    cols_vals = valid["Position"].apply(lambda p: p[1])
    grid_h = int(rows_vals.max()) + 1
    grid_w = int(cols_vals.max()) + 1
    grid = np.zeros((grid_h, grid_w))
    for r, c in zip(rows_vals, cols_vals):
        grid[int(r), int(c)] += 1
    render_heatmap(ax, grid, "Visitation Heatmap")


@register_plot("Action Distribution")
def plot_action_distribution(ax, df, metric, groups, group_col):
    """Normalized action frequency bar chart across all episodes in the selected block."""
    ax.clear()
    valid = df[df["Action"].apply(lambda x: x is not None)]
    if valid.empty:
        return
    all_actions = sorted(valid["Action"].unique())
    counts = valid["Action"].value_counts().reindex(all_actions, fill_value=0)
    total = counts.sum()
    fractions = (counts / total).values if total > 0 else counts.values
    render_bar(
        ax,
        [str(a) for a in all_actions],
        fractions,
        "Action Distribution",
        get_color(0),
    )


@register_plot("ROI Turn Distribution")
def plot_roi_turn_distribution(ax, df, metric, groups, group_col):
    """Normalized internal action (ROI turn) frequency bar chart."""
    ax.clear()
    valid = df[df["Internal_action"].notna()]
    if valid.empty:
        return
    all_actions = sorted(valid["Internal_action"].unique())
    counts = valid["Internal_action"].value_counts().reindex(all_actions, fill_value=0)
    total = counts.sum()
    fractions = (counts / total).values if total > 0 else counts.values
    render_bar(
        ax,
        [str(int(a)) for a in all_actions],
        fractions,
        "ROI Turn Distribution",
        get_color(0),
    )


@register_plot("Optimality Scatter")
def plot_optimality_scatter(ax, df, metric, groups, group_col):
    """Steps to goal vs spawn distance (Path Optimality Index). Points on y=x achieved POI=1."""
    ax.clear()
    episodes = per_episode_efficiency(df)
    reached = [e for e in episodes if e["steps_to_goal"] != float("inf")]
    if not reached:
        return
    xs = [e["spawn_dist"] for e in reached]
    ys = [e["steps_to_goal"] for e in reached]
    render_scatter(ax, xs, ys, "Optimality Scatter")
