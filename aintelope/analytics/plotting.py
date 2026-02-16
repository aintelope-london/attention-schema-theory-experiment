"""Plotting toolkit for experiment results visualization.

All matplotlib usage is centralized here. Consumers receive or pass
figure/axes objects — they never import matplotlib directly.
"""

from matplotlib.figure import Figure

# =============================================================================
# Plot registry
# =============================================================================

PLOT_TYPES = {}


def register_plot(name):
    """Decorator to register a plot function by name."""

    def decorator(fn):
        PLOT_TYPES[name] = fn
        return fn

    return decorator


# =============================================================================
# Color palette
# =============================================================================

PALETTE = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
]


def get_color(index):
    """Return a color from the palette, cycling if needed."""
    return PALETTE[index % len(PALETTE)]


# =============================================================================
# Figure management
# =============================================================================


def create_figure(figsize=(10, 6)):
    """Create a new Figure with a single subplot."""
    figure = Figure(figsize=figsize)
    ax = figure.add_subplot(111)
    return figure, ax


def save_figure(figure, path):
    """Save figure to file."""
    figure.savefig(path, bbox_inches="tight")


# =============================================================================
# Data aggregation primitives
# =============================================================================


def collapse(df, group_cols, y_col, fn="sum"):
    """Reduce rows by grouping and aggregating.

    Args:
        df: Source DataFrame.
        group_cols: Columns to group by.
        y_col: Column to aggregate.
        fn: Aggregation function name.

    Returns:
        DataFrame with one row per group, y_col aggregated.
    """
    return df.groupby(group_cols, as_index=False)[y_col].agg(fn)


def aggregate_series(df, x_col, y_col):
    """Compute mean and std of y_col grouped by x_col.

    Args:
        df: Source DataFrame.
        x_col: Column for the x-axis.
        y_col: Column to compute statistics on.

    Returns:
        (x, mean, std) as numpy arrays.
    """
    agg = df.groupby(x_col)[y_col].agg(["mean", "std"]).fillna(0)
    return agg.index.values, agg["mean"].values, agg["std"].values


# =============================================================================
# Rendering primitives
# =============================================================================

''' TODO clenaup
def plot_band(ax, x, mean, std, label=None, color=None, alpha=0.2):
    """Draw a line with shaded standard deviation region.

    Args:
        ax: Matplotlib axes.
        x: X-axis values.
        mean: Mean values.
        std: Standard deviation values.
        label: Legend label.
        color: Line and fill color.
        alpha: Fill transparency.
    """
    ax.plot(x, mean, linewidth=0.75, label=label, color=color)
    ax.fill_between(x, mean - std, mean + std, alpha=alpha, color=color)
'''


def plot_band(ax, x, mean, std, label=None, color=None, alpha=0.15):
    ax.plot(x, mean, linewidth=1.5, label=label, color=color)
    ax.fill_between(x, mean - std, mean + std, alpha=alpha, color=color, linewidth=0)


def plot_grouped_bands(ax, df, metric, groups, group_col, series_fn, x_label):
    """Render a band per group, each in its own color.

    Args:
        ax: Matplotlib axes.
        df: Source DataFrame.
        metric: Column name for the y-axis value.
        groups: List of group values to iterate over.
        group_col: Column name to filter by.
        series_fn: Callable(df, metric) -> (x, mean, std).
        x_label: Label for the x-axis.
    """
    ax.clear()
    for i, group in enumerate(groups):
        group_df = df[df[group_col] == group]
        x, mean, std = series_fn(group_df, metric)
        plot_band(ax, x, mean, std, label=group, color=get_color(i))
    ax.set_xlabel(x_label)
    ax.set_ylabel(metric)
    if len(groups) > 1:
        ax.legend()
    ax.figure.tight_layout()
