# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Core analytics functions for evaluating learning outcomes."""

import pandas as pd
from aintelope.analytics.diagnostics import compute_learning_analytics


# TODO DEPRECATED — use compute_learning_analytics + assert_learning_improvement
def assert_learning_threshold(
    events: pd.DataFrame, threshold: float, phase: str = "test"
) -> bool:
    """Check if average episode reward exceeds threshold."""
    phase_events = events[events["IsTest"]] if phase == "test" else events[~events["IsTest"]]
    episode_rewards = phase_events.groupby("Episode")["Reward"].sum()
    avg_reward = episode_rewards.mean()
    assert avg_reward >= threshold, (
        f"Learning failed: {phase} avg reward {avg_reward:.3f} < threshold {threshold:.3f}"
    )
    return True


def assert_learning_improvement(
    analytics: dict,
    phase: str = "train",
) -> None:
    """Assert learning improvement from a pre-computed analytics dict.

    Args:
        analytics: dict returned by compute_learning_analytics()
        phase: which phase to assert on — 'train' or 'test'

    Raises:
        AssertionError if improvement criterion not met
    """
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


def calculate_optimal_steps(agent_positions: list, food_positions: list) -> list:
    """Stub for future implementation."""
    raise NotImplementedError("Optimal steps calculation not yet implemented")