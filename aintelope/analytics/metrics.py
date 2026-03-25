# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Metric computation for experiment results.

Each function receives the full results dict and params, computes its metric
across all blocks, and returns {block: data_dict}. No rendering or file I/O.

results shape: {block_name: {events, states, learning_df, manifesto, cfg}}
"""

import numpy as np
import pandas as pd

from aintelope.analytics.diagnostics import collector
from aintelope.analytics.plot_primitives import aggregate_series, collapse
from aintelope.analytics.recording import SERIALIZABLE_COLUMNS, deserialize_state


# ── Helpers ───────────────────────────────────────────────────────────────────


def text(title, rows):
    bar = "─" * 50
    header = f"── {title} " + bar[len(title) + 4 :]
    return "\n".join([header] + rows)


def filter_events(events, **kwargs):
    """Filter events by column equality. Returns a copy; no deserialization."""
    mask = pd.Series(True, index=events.index)
    for col, val in kwargs.items():
        mask &= events[col] == val
    return events[mask].copy()


def deserialize_events(events):
    """Deserialize SERIALIZABLE_COLUMNS in-place on a copy. Call only when
    observation data is needed and the DataFrame still holds raw CSV strings."""
    result = events.copy()
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


def _dist_to_nearest(vision, channel):
    h, w = vision.shape[1], vision.shape[2]
    cr, cc = h // 2, w // 2
    coords = list(zip(*np.where(vision[channel] > 0)))
    if not coords:
        return None
    return min(abs(r - cr) + abs(c - cc) for r, c in coords)


def per_episode_efficiency(events):
    """Compute per-episode optimality ratio (spawn_dist / steps_to_goal).

    spawn_dist is the Manhattan distance at episode start — the theoretical
    minimum steps to goal, i.e. the Path Optimality Index (POI) denominator.

    Returns list of {trial, episode, spawn_dist, steps_to_goal, efficiency}.
    """
    step0 = filter_events(events, Step=0)
    first = first_reward(events).set_index(["Episode", "Trial"])["steps_to_reward"]
    out = []
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
        out.append(
            {
                "trial": int(trial),
                "episode": int(episode),
                "spawn_dist": spawn_dist,
                "steps_to_goal": steps_to_goal,
                "efficiency": efficiency,
            }
        )
    return out


# ── Metric functions ──────────────────────────────────────────────────────────


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
    out = {}
    for block, data in results.items():
        collapsed = collapse(data["events"], ["Episode", "Trial"], "Reward", "sum")
        out[block] = {"series": aggregate_series(collapsed, "Episode", "Reward")}
    return out


def loss_curve(results, params):
    out = {}
    for block, data in results.items():
        df = data["learning_df"]
        if df.empty or "loss" not in df.columns:
            continue
        collapsed = collapse(df, ["episode", "trial"], "loss", "mean")
        collapsed.columns = ["Episode", "Trial", "Loss"]
        out[block] = {"series": aggregate_series(collapsed, "Episode", "Loss")}
    return out


def epsilon_curve(results, params):
    out = {}
    for block, data in results.items():
        df = data["learning_df"]
        if df.empty or "epsilon" not in df.columns:
            continue
        collapsed = collapse(df, ["episode", "trial"], "epsilon", "mean")
        collapsed.columns = ["Episode", "Trial", "Epsilon"]
        out[block] = {"series": aggregate_series(collapsed, "Episode", "Epsilon")}
    return out


def reward_curve(results, params):
    out = {}
    for block, data in results.items():
        df = data["learning_df"]
        if df.empty or "reward" not in df.columns:
            continue
        collapsed = collapse(df, ["episode", "trial"], "reward", "mean")
        collapsed.columns = ["Episode", "Trial", "Reward"]
        out[block] = {"series": aggregate_series(collapsed, "Episode", "Reward")}
    return out


def steps_to_reward(results, params):
    out = {}
    for block, data in results.items():
        steps_df = first_reward(data["events"])
        if steps_df.empty:
            continue
        out[block] = {
            "series": aggregate_series(steps_df, "Episode", "steps_to_reward")
        }
    return out


def optimal_efficiency(results, params):
    min_efficiency_pct = params.get("min_efficiency_pct", 0.70) * 100
    out = {}
    report_lines = []
    for block, data in results.items():
        episodes = per_episode_efficiency(data["events"])
        valid = [e["efficiency"] for e in episodes if e["efficiency"] is not None]
        mean_eff = float(np.mean(valid)) * 100 if valid else 0.0
        out[block] = {
            "efficiency_pct": mean_eff,
            "per_episode": episodes,
            "n_episodes": len(episodes),
            "min_efficiency_pct": min_efficiency_pct,
        }
        report_lines.append(f"Block: {block}")
        for ep in episodes:
            dist = ep["spawn_dist"] if ep["spawn_dist"] is not None else "?"
            steps = ep["steps_to_goal"] if ep["steps_to_goal"] is not None else "never"
            eff = (
                f"{ep['efficiency'] * 100:.0f}%"
                if ep["efficiency"] is not None
                else "N/A"
            )
            report_lines.append(
                f"  Trial {ep['trial']:>2}, Episode {ep['episode']:>4}: "
                f"spawn_dist={dist}, steps_to_goal={steps}, efficiency={eff}"
            )
        eff_str = f"{mean_eff:.1f}%" if mean_eff is not None else "N/A"
        report_lines.append(
            f"  Efficiency: {eff_str} (mean over {len(episodes)} episodes)"
        )
        report_lines.append("")
    collector.collect(
        {"Optimal Policy Report": text("Optimal Policy Report", report_lines)}
    )
    return out


def efficiency_curve(results, params):
    out = {}
    for block, data in results.items():
        episodes = per_episode_efficiency(data["events"])
        df = pd.DataFrame(episodes).rename(
            columns={"episode": "Episode", "efficiency": "Efficiency"}
        )
        out[block] = {"series": aggregate_series(df, "Episode", "Efficiency")}
    return out


def visitation_heatmap(results, params):
    out = {}
    for block, data in results.items():
        events = data["events"]
        valid = events[events["Position"].apply(lambda x: x is not None)]
        out[block] = {
            "events": valid,
            "n_windows": params.get("n_windows", 2),
            "block": block,
        }
    return out


def action_distribution(results, params):
    out = {}
    for block, data in results.items():
        events = data["events"]
        valid = events[events["Action"].apply(lambda x: x is not None)]
        action_names = data["manifesto"].get("action_names", {})
        out[block] = {
            "events": valid,
            "action_names": action_names,
            "n_windows": params.get("n_windows", 2),
            "block": block,
        }
    return out


def roi_turn_distribution(results, params):
    out = {}
    for block, data in results.items():
        events = data["events"]
        valid = events[events["Internal_action"].notna()]
        out[block] = {
            "events": valid,
            "n_windows": params.get("n_windows", 2),
            "block": block,
        }
    return out


def roi_food_alignment(results, params):
    out = {}
    for block, data in results.items():
        events = deserialize_events(data["events"])
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
        out[block] = {"series": aggregate_series(df, "Episode", "food_in_roi")}
    return out


_METRICS = {
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
