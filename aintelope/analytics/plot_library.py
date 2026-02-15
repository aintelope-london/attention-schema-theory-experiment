"""Plot library — registered plot types for publication figures.

Each function composes toolkit primitives from plotting.py.
"""

from aintelope.analytics.plotting import (
    register_plot,
    collapse,
    aggregate_series,
    plot_grouped_bands,
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
