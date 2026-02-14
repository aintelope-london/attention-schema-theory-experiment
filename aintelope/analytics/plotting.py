"""Plotting functions for experiment results visualization.

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
# Data preparation
# =============================================================================


def prepare_plot_data(df, filter_by=None, group_by=None, agg="mean"):
    """Filter and aggregate a DataFrame for plotting.

    Args:
        df: Source DataFrame.
        filter_by: Dict of {column: value} to filter rows.
        group_by: List of columns to group by before aggregating.
        agg: Aggregation function name (e.g. "mean", "sum", "median").

    Returns:
        Transformed DataFrame ready for plotting.
    """
    result = df
    if filter_by:
        for col, val in filter_by.items():
            result = result[result[col] == val]
    if group_by:
        result = result.groupby(group_by, as_index=False).agg(agg)
    return result


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
# Plot types
# =============================================================================


@register_plot("line")
def plot_line(ax, df, x_col, y_cols):
    """Line plot of one or more y columns against x.

    Args:
        ax: Matplotlib Axes to draw on.
        df: DataFrame with the data.
        x_col: Column name for x axis.
        y_cols: List of column names for y axis.
    """
    ax.clear()
    for col in y_cols:
        ax.plot(df[x_col], df[col], linewidth=0.75, label=col)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_cols[0] if len(y_cols) == 1 else "Value")
    if len(y_cols) > 1:
        ax.legend()
    ax.figure.tight_layout()
