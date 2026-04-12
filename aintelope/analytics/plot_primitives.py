# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Plotting primitives for experiment result visualization.

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
    return PALETTE[index % len(PALETTE)]


# =============================================================================
# Figure management
# =============================================================================


def create_figure(figsize=(10, 6)):
    figure = Figure(figsize=figsize)
    ax = figure.add_subplot(111)
    return figure, ax


def create_figure_grid(ncols, figsize=None):
    """Create a Figure with one row of ncols subplots. Returns (figure, axes list)."""
    figsize = figsize or (5 * ncols, 4)
    figure = Figure(figsize=figsize)
    return figure, [figure.add_subplot(1, ncols, i + 1) for i in range(ncols)]


def save_figure(figure, path):
    figure.savefig(path, bbox_inches="tight")


# =============================================================================
# Data aggregation
# =============================================================================


def collapse(df, group_cols, y_col, fn="sum"):
    return df.groupby(group_cols, as_index=False).agg({y_col: fn})


def aggregate_series(df, x_col, y_col):
    agg = df.groupby(x_col)[y_col].agg(["mean", "std"]).fillna(0)
    return (
        agg.index.values.astype(float),
        agg["mean"].values.astype(float),
        agg["std"].values.astype(float),
    )


# =============================================================================
# Rendering primitives
# =============================================================================


def plot_band(ax, x, mean, std, label=None, color=None, alpha=0.15):
    ax.plot(x, mean, linewidth=1.5, label=label, color=color)
    ax.fill_between(x, mean - std, mean + std, alpha=alpha, color=color, linewidth=0)


def plot_grouped_bands(ax, df, metric, groups, group_col, series_fn, x_label):
    """Render one band per group into ax."""
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


def render_heatmap(ax, grid, title):
    """2D count grid rendered as a heatmap."""
    im = ax.imshow(grid, origin="upper", cmap="hot")
    ax.set_title(title)
    ax.set_xlabel("col")
    ax.set_ylabel("row")
    ax.figure.colorbar(im, ax=ax)


def render_bar(ax, labels, values, title, color):
    """Normalized bar chart for categorical fractions."""
    ax.bar(labels, values, color=color)
    ax.set_title(title)
    ax.set_xlabel("Action")
    ax.set_ylabel("Fraction")
    ax.set_ylim(0, 1)


def render_scatter(ax, xs, ys, title):
    """Scatter of actual vs optimal steps. The diagonal y=x is the optimality frontier."""
    ax.scatter(xs, ys, alpha=0.6, s=20, color=get_color(0))
    if xs:
        max_val = max(max(xs), max(ys))
        ax.plot(
            [0, max_val],
            [0, max_val],
            "--",
            color="grey",
            linewidth=1.2,
            label="optimal (POI=1)",
        )
    ax.set_xlabel("Optimal steps (spawn distance)")
    ax.set_ylabel("Actual steps to goal")
    ax.set_title(title)
    ax.legend()
    ax.figure.tight_layout()


def plot_series(
    series_by_label, x_label, y_label, title, ref_line=None, yscale="linear"
):
    """Create a multi-series band chart figure."""
    figure, ax = create_figure()
    for i, (label, (x, mean, std)) in enumerate(series_by_label.items()):
        plot_band(ax, x, mean, std, label=label, color=get_color(i))
    if ref_line is not None:
        ref_label, ref_y = ref_line
        ax.axhline(ref_y, linestyle="--", color="grey", linewidth=1.2, label=ref_label)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_yscale(yscale)
    if series_by_label or ref_line:
        ax.legend()
    ax.figure.tight_layout()
    return figure
