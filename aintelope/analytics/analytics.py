# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Analytics — single entry point for all run diagnostics.

analyze(results) iterates cfg.run.analytics, calls each library function,
and returns {name: {block: result}} for test assertions.

Each library function receives the full results dict, computes its analytic,
writes its own outputs (figures, CSVs), and sends text sections to
diagnostics.collector. Returns computed data keyed by block name.

results shape: {block_name: {events, states, learning_df, manifesto, cfg}}
"""

from pathlib import Path

import numpy as np
import pandas as pd

from aintelope.analytics.diagnostics import collector
from aintelope.analytics.plotting import (
    aggregate_series,
    collapse,
    create_figure,
    create_figure_grid,
    get_color,
    plot_band,
    render_bar,
    render_heatmap,
    save_figure,
)
from aintelope.analytics.recording import (
    write_csv,
    SERIALIZABLE_COLUMNS,
    deserialize_state,
)


# ── Render helpers ────────────────────────────────────────────────────────────


def plot(series_by_label, x_label, y_label, title, ref_line=None, yscale="linear"):
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


def text(title, rows):
    """Formatted text block for report sections."""
    bar = "─" * 50
    header = f"── {title} " + bar[len(title) + 4 :]
    return "\n".join([header] + rows)


# ── Output helpers ────────────────────────────────────────────────────────────


def _cfg(results):
    return next(iter(results.values()))["cfg"]


def _write_figure(results, name, fig):
    cfg = _cfg(results)
    if cfg.run.write_outputs:
        path = Path(cfg.run.outputs_dir) / f"{name}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        save_figure(fig, path)


def _write_df(results, name, df):
    cfg = _cfg(results)
    if cfg.run.write_outputs and not df.empty:
        write_csv(Path(cfg.run.outputs_dir) / f"{name}.csv", df)


# ── Preprocessing ─────────────────────────────────────────────────────────────


def filter_events(events, **kwargs):
    """Filter events and deserialize observation columns.

    Deserialization is deferred to here — call selectively to avoid loading
    all observations into memory.
    """
    mask = pd.Series(True, index=events.index)
    for col, val in kwargs.items():
        mask &= events[col] == val
    result = events[mask].copy()
    for col in SERIALIZABLE_COLUMNS:
        if col in result.columns:
            result[col] = result[col].apply(
                lambda x: deserialize_state(x) if x is not None else None
            )
    return result


def first_reward(events):
    """Per-episode step index of first positive reward.

    Returns DataFrame[Episode, Trial, steps_to_reward].
    """
    positive = events[events["Reward"] > 0][["Episode", "Trial", "Step"]]
    if positive.empty:
        return pd.DataFrame(columns=["Episode", "Trial", "steps_to_reward"])
    first = positive.groupby(["Episode", "Trial"])["Step"].min().reset_index()
    first.columns = ["Episode", "Trial", "steps_to_reward"]
    return first


def _episode_windows(episodes, n_windows):
    eps = sorted(set(episodes))
    n_windows = min(n_windows, len(eps))
    size = max(1, len(eps) // n_windows)
    windows = []
    for i in range(n_windows):
        start = i * size
        end = start + size if i < n_windows - 1 else len(eps)
        label = f"ep {eps[start]}–{eps[end - 1]}"
        windows.append((label, frozenset(eps[start:end])))
    return windows


# ── Compute ───────────────────────────────────────────────────────────────────


def _dist_to_nearest(vision, channel):
    h, w = vision.shape[1], vision.shape[2]
    cr, cc = h // 2, w // 2
    coords = list(zip(*np.where(vision[channel] > 0)))
    if not coords:
        return None
    return min(abs(r - cr) + abs(c - cc) for r, c in coords)


def _per_episode_efficiency(events, food_ind):
    step0 = filter_events(events, Step=0)
    first = first_reward(events).set_index(["Episode", "Trial"])["steps_to_reward"]

    per_episode = []
    for (episode, trial), group in step0.groupby(["Episode", "Trial"]):
        row = group.iloc[0]
        spawn_dist = abs(row["Position"][0] - row["Food_position"][0]) + abs(
            row["Position"][1] - row["Food_position"][1]
        )
        steps_to_goal = (
            int(first.loc[(episode, trial)]) + 1
            if (episode, trial) in first.index
            else float("inf")
        )
        efficiency = (
            1.0
            if spawn_dist == 0
            else 0.0
            if steps_to_goal == float("inf")
            else min(1.0, spawn_dist / steps_to_goal)
        )
        per_episode.append(
            {
                "trial": int(trial),
                "episode": int(episode),
                "spawn_dist": spawn_dist,
                "steps_to_goal": steps_to_goal,
                "efficiency": efficiency,
            }
        )
    return per_episode


# ── Entry point ───────────────────────────────────────────────────────────────


def analyze(results):
    """Run configured analytics. Returns {name: {block: result}}."""
    cfg = _cfg(results)
    analytics = {}
    for name, params in cfg.run.analytics.items():
        analytics[name] = _ANALYTICS[name](results, params)
    return analytics


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
    is already in report.txt via optimal_efficiency analytic.
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


# ── Library ───────────────────────────────────────────────────────────────────


def _format_architecture(architecture):
    parts = []
    for cid, entry in architecture.items():
        inputs = ",".join(entry.get("inputs", []))
        parts.append(f"{cid}={entry['type']}({inputs})")
    return " | ".join(parts)


def run_summary(results, params):
    out = {}
    lines_all = []
    for block, data in results.items():
        cfg = data["cfg"]
        agent_cfg = cfg.agent_params.agent_0
        lines = [f"Block:    {block}", f"Agent:    {agent_cfg.agent_class}"]
        if hasattr(agent_cfg, "architecture"):
            arch = {
                cid: {"type": e.type, "inputs": list(e.inputs)}
                for cid, e in agent_cfg.architecture.items()
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
        train_parts = [
            f"{lbl}={getattr(cfg.agent_params, attr)}"
            for attr, lbl in [
                ("gamma", "γ"),
                ("batch_size", "batch"),
                ("learning_rate", "lr"),
            ]
            if hasattr(cfg.agent_params, attr)
        ]
        if train_parts:
            lines.append(f"Training: {' | '.join(train_parts)}")
        manifesto = data.get("manifesto", {})
        if manifesto:
            skip = {"observation_shapes", "action_space"}
            m_parts = [f"{k}={v}" for k, v in manifesto.items() if k not in skip]
            lines.append(f"Manifesto: {' | '.join(m_parts)}")
        lines_all += lines + [""]
        out[block] = lines
    collector.collect({"Run Summary": text("Run Summary", lines_all)})
    return out


def learning_improvement(results, params):
    episode_fraction = params.get("episode_fraction", 0.15)
    min_improvement_ratio = params.get("min_improvement_ratio", 1.3)
    out = {}
    lines = []
    for block, data in results.items():
        ep_rewards = data["events"].groupby("Episode")["Reward"].sum().sort_index()
        n = len(ep_rewards)
        window = max(1, int(n * episode_fraction))
        start_avg = float(ep_rewards.iloc[:window].mean())
        end_avg = float(ep_rewards.iloc[-window:].mean())
        ratio = (end_avg / start_avg) if start_avg > 0 else None
        passed = (
            (end_avg > start_avg) if ratio is None else (ratio >= min_improvement_ratio)
        )
        block_result = {
            "ratio": ratio,
            "start_avg": start_avg,
            "end_avg": end_avg,
            "window": window,
            "passed": passed,
            "min_improvement_ratio": min_improvement_ratio,
        }
        status = "✓" if passed else "✗"
        lines.append(
            f"  {block}: {status}  ratio={ratio:.2f}x  (start={start_avg:.3f}, end={end_avg:.3f}, window={window} eps)"
            if ratio is not None
            else f"  {block}: {status}  start={start_avg:.3f} → end={end_avg:.3f}"
        )
        out[block] = block_result
    collector.collect({"Learning Improvement": text("Learning Improvement", lines)})
    return out


def learning_curve(results, params):
    series = {}
    for block, data in results.items():
        collapsed = collapse(data["events"], ["Episode", "Trial"], "Reward", "sum")
        series[block] = aggregate_series(collapsed, "Episode", "Reward")
    fig = plot(
        series, x_label="Episode", y_label="Total Reward", title="Learning Curve"
    )
    _write_figure(results, "learning_curve", fig)
    return {"figure": fig}


def loss_curve(results, params):
    series = {}
    for block, data in results.items():
        df = data["learning_df"]
        if df.empty or "loss" not in df.columns:
            continue
        collapsed = collapse(df, ["episode", "trial"], "loss", "mean")
        collapsed.columns = ["Episode", "Trial", "Loss"]
        series[block] = aggregate_series(collapsed, "Episode", "Loss")
    if not series:
        return {}
    fig = plot(series, x_label="Episode", y_label="Mean Loss", title="Loss Curve")
    _write_figure(results, "loss_curve", fig)
    return {"figure": fig}


def epsilon_curve(results, params):
    series = {}
    for block, data in results.items():
        df = data["learning_df"]
        if df.empty or "epsilon" not in df.columns:
            continue
        collapsed = collapse(df, ["episode", "trial"], "epsilon", "mean")
        collapsed.columns = ["Episode", "Trial", "Epsilon"]
        series[block] = aggregate_series(collapsed, "Episode", "Epsilon")
    if not series:
        return {}
    fig = plot(series, x_label="Episode", y_label="Epsilon", title="Epsilon Decay")
    _write_figure(results, "epsilon_curve", fig)
    return {"figure": fig}


def reward_curve(results, params):
    series = {}
    for block, data in results.items():
        df = data["learning_df"]
        if df.empty or "reward" not in df.columns:
            continue
        collapsed = collapse(df, ["episode", "trial"], "reward", "mean")
        collapsed.columns = ["Episode", "Trial", "Reward"]
        series[block] = aggregate_series(collapsed, "Episode", "Reward")
    if not series:
        return {}
    fig = plot(series, x_label="Episode", y_label="Mean Reward", title="Reward Signal")
    _write_figure(results, "reward_curve", fig)
    return {"figure": fig}


def steps_to_reward(results, params):
    series = {}
    for block, data in results.items():
        steps_df = first_reward(data["events"])
        if steps_df.empty:
            continue
        series[block] = aggregate_series(steps_df, "Episode", "steps_to_reward")
    if not series:
        return {}
    fig = plot(
        series,
        x_label="Episode",
        y_label="Steps to First Reward",
        title="Steps to Reward",
    )
    _write_figure(results, "steps_to_reward", fig)
    return {"figure": fig}


def optimal_efficiency(results, params):
    min_efficiency_pct = params.get("min_efficiency_pct", 0.70) * 100
    out = {}
    report_lines = []
    for block, data in results.items():
        per_episode = _per_episode_efficiency(
            data["events"], data["manifesto"]["food_ind"]
        )
        valid = [e["efficiency"] for e in per_episode if e["efficiency"] is not None]
        mean_eff = float(np.mean(valid)) * 100 if valid else 0.0
        out[block] = {
            "efficiency_pct": mean_eff,
            "per_episode": per_episode,
            "n_episodes": len(per_episode),
            "min_efficiency_pct": min_efficiency_pct,
        }
        report_lines.append(f"Block: {block}")
        for ep in per_episode:
            dist = ep["spawn_dist"] if ep["spawn_dist"] is not None else "?"
            steps = ep["steps_to_goal"] if ep["steps_to_goal"] is not None else "never"
            eff = (
                f"{ep['efficiency'] * 100:.0f}%"
                if ep["efficiency"] is not None
                else "N/A"
            )
            report_lines.append(
                f"  Trial {ep['trial']:>2}, Episode {ep['episode']:>4}: spawn_dist={dist}, steps_to_goal={steps}, efficiency={eff}"
            )
        n = len(per_episode)
        eff_str = f"{mean_eff:.1f}%" if mean_eff is not None else "N/A"
        report_lines.append(f"  Efficiency: {eff_str} (mean over {n} episodes)")
        report_lines.append("")
    collector.collect(
        {"Optimal Policy Report": text("Optimal Policy Report", report_lines)}
    )
    return out


def efficiency_curve(results, params):
    series = {}
    for block, data in results.items():
        per_episode = _per_episode_efficiency(
            data["events"], data["manifesto"]["food_ind"]
        )
        df = pd.DataFrame(per_episode).rename(
            columns={"episode": "Episode", "efficiency": "Efficiency"}
        )
        series[block] = aggregate_series(df, "Episode", "Efficiency")
    if not series:
        return {}
    fig = plot(
        series,
        x_label="Episode",
        y_label="Efficiency",
        title="Policy Efficiency over Training",
    )
    _write_figure(results, "efficiency_curve", fig)
    return {"figure": fig}


def visitation_heatmap(results, params):
    n_windows = params.get("n_windows", 2)
    out = {}
    for block, data in results.items():
        events = data["events"]
        valid = events[events["Position"].apply(lambda x: x is not None)]
        if valid.empty:
            continue
        rows = valid["Position"].apply(lambda p: p[0])
        cols = valid["Position"].apply(lambda p: p[1])
        grid_h, grid_w = int(rows.max()) + 1, int(cols.max()) + 1
        windows = _episode_windows(valid["Episode"].unique(), n_windows)
        figure, axes = create_figure_grid(n_windows)
        for ax, (label, ep_set) in zip(axes, windows):
            mask = valid["Episode"].isin(ep_set)
            grid = np.zeros((grid_h, grid_w))
            for r, c in zip(rows[mask], cols[mask]):
                grid[int(r), int(c)] += 1
            render_heatmap(ax, grid, f"{block} — {label}")
        figure.tight_layout()
        _write_figure(results, f"visitation_heatmap_{block}", figure)
        out[block] = {"figure": figure}
    return out


def action_distribution(results, params):
    n_windows = params.get("n_windows", 2)
    out = {}
    for block, data in results.items():
        events = data["events"]
        valid = events[events["Action"].apply(lambda x: x is not None)]
        if valid.empty:
            continue
        all_actions = sorted(valid["Action"].unique())
        action_names = data["manifesto"].get("action_names", {})
        labels = [action_names.get(a, str(a)) for a in all_actions]
        windows = _episode_windows(valid["Episode"].unique(), n_windows)
        figure, axes = create_figure_grid(n_windows)
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
        _write_figure(results, f"action_distribution_{block}", figure)
        out[block] = {"figure": figure}
    return out


def roi_turn_distribution(results, params):
    n_windows = params.get("n_windows", 2)
    out = {}
    for block, data in results.items():
        events = data["events"]
        valid = events[events["Internal_action"].notna()]
        if valid.empty:
            continue
        all_actions = sorted(valid["Internal_action"].unique())
        labels = [str(int(a)) for a in all_actions]
        windows = _episode_windows(valid["Episode"].unique(), n_windows)
        figure, axes = create_figure_grid(n_windows)
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
        _write_figure(results, f"roi_turn_distribution_{block}", figure)
        out[block] = {"figure": figure}
    return out


def roi_food_alignment(results, params):
    series = {}
    for block, data in results.items():
        events = filter_events(data["events"])
        food_ind = data["manifesto"]["food_ind"]
        valid = events[events["Observation"].apply(lambda x: x is not None)]
        if valid.empty:
            continue
        df = valid.copy()
        df["food_in_roi"] = df["Observation"].apply(
            lambda obs: bool(
                np.any((obs["vision"][food_ind] > 0) & (obs["vision"][-1] > 0))
            )
        )
        series[block] = aggregate_series(df, "Episode", "food_in_roi")
    if not series:
        return {}
    fig = plot(
        series,
        x_label="Episode",
        y_label="Food in ROI rate",
        title="ROI Food Alignment over Training",
    )
    _write_figure(results, "roi_food_alignment", fig)
    return {"figure": fig}


_ANALYTICS = {
    "run_summary": run_summary,
    "learning_improvement": learning_improvement,
    "learning_curve": learning_curve,
    "loss_curve": loss_curve,
    "epsilon_curve": epsilon_curve,
    "reward_curve": reward_curve,
    "steps_to_reward": steps_to_reward,
    "optimal_efficiency": optimal_efficiency,
    "efficiency_curve": efficiency_curve,
    "visitation_heatmap": visitation_heatmap,
    "action_distribution": action_distribution,
    "roi_turn_distribution": roi_turn_distribution,
    "roi_food_alignment": roi_food_alignment,
}
