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


def beelines_to_object(events, states, agent_id, target_position):
    """Per-episode optimality: steps to reach nearest target vs Manhattan spawn distance.

    For each episode:
      1. Spawn position: read from states at step -1, extract agent layer from board cube.
      2. Target coordinate: read from first event row (retained from episode start).
      3. Find first step where agent Position matches that retained coordinate.
      4. efficiency = spawn_dist / steps_to_reach.

    The target coordinate is retained from episode start — the tile may disappear
    after being reached, but the coordinate persists within this computation.

    Args:
        events:          events DataFrame
        states:          states DataFrame (must include step -1 reset rows)
        agent_id:        filters Agent_id column; also identifies agent board layer
        target_position: column name holding target coordinate(s) per step.
                         Accepts a single (row, col) tuple or a list of tuples
                         for future multi-object columns. Always uses nearest.

    Returns list of {trial, episode, spawn_dist, steps_to_reach, efficiency}.
    """
    agent_events = events[events["Agent_id"] == agent_id]
    out = []
    for (trial, episode), group in agent_events.groupby(["Trial", "Episode"]):
        group = group.sort_values("Step")
        row0 = group.iloc[0]

        # Agent spawn position from reset state at step -1.
        spawn_row = states[
            (states["Trial"] == trial)
            & (states["Episode"] == episode)
            & (states["Step"] == -1)
        ]
        if spawn_row.empty:
            continue
        state = deserialize_state(spawn_row["Board"].iloc[0])
        board_cube, layers = state["board"], state["layers"]
        agent_layer = layers.index(agent_id)
        ys, xs = np.where(board_cube[agent_layer] > 0)
        if len(ys) == 0:
            continue
        agent_pos = (int(ys[0]), int(xs[0]))

        target_val = row0[target_position]
        if target_val is None:
            continue

        # Normalize to list — ready for future multi-object columns
        targets = target_val if isinstance(target_val, list) else [target_val]

        # Retain nearest target coordinate and spawn distance
        spawn_dist, nearest = min(
            ((abs(agent_pos[0] - t[0]) + abs(agent_pos[1] - t[1])), t) for t in targets
        )

        # Find first step where agent position matches the retained coordinate
        reached = group[group["Position"].apply(lambda p: p == nearest)]
        steps_to_reach = (
            int(reached["Step"].iloc[0]) + 1 if not reached.empty else float("inf")
        )
        efficiency = (
            1.0
            if spawn_dist == 0
            else 0.0
            if steps_to_reach == float("inf")
            else min(1.0, spawn_dist / steps_to_reach)
        )
        out.append(
            {
                "trial": int(trial),
                "episode": int(episode),
                "spawn_dist": spawn_dist,
                "steps_to_reach": steps_to_reach,
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
        agent_cfg = cfg.agent_params.agents.agent_0
        lines = [f"Block:    {block}", f"Agent:    {agent_cfg.agent_class}"]
        if hasattr(agent_cfg, "architecture"):
            arch = {
                cid: {"type": e.type, "inputs": list(e.inputs)}
                for cid, e in agent_cfg.architecture.items()
            }
            lines.append(f"Arch:     {_format_architecture(arch)}")
        env_keys = (
            "map_size",
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
        events = data["events"]
        agent_id = params.get("agent_id", events["Agent_id"].iloc[0])
        episodes = beelines_to_object(events, data["states"], agent_id, "Food_position")
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
            dist = ep["spawn_dist"]
            steps = (
                ep["steps_to_reach"]
                if ep["steps_to_reach"] != float("inf")
                else "never"
            )
            eff = f"{ep['efficiency'] * 100:.0f}%"
            report_lines.append(
                f"  Trial {ep['trial']:>2}, Episode {ep['episode']:>4}: "
                f"spawn_dist={dist}, steps_to_reach={steps}, efficiency={eff}"
            )
        report_lines.append(
            f"  Efficiency: {mean_eff:.1f}% (mean over {len(episodes)} episodes)"
        )
        report_lines.append("")
    collector.collect(
        {"Optimal Policy Report": text("Optimal Policy Report", report_lines)}
    )
    return out


def efficiency_curve(results, params):
    out = {}
    for block, data in results.items():
        events = data["events"]
        agent_id = params.get("agent_id", events["Agent_id"].iloc[0])
        episodes = beelines_to_object(events, data["states"], agent_id, "Food_position")
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
