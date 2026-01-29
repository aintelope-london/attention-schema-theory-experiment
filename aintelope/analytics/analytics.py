# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Core analytics functions for evaluating learning outcomes."""


def assert_learning_threshold(test_summary: dict, threshold: float) -> bool:
    """Check if average test reward exceeds threshold.

    Args:
        test_summary: Output from pipeline analytics containing test_averages
        threshold: Minimum average reward to consider learning successful

    Returns:
        True if learning threshold met

    Raises:
        AssertionError if threshold not met
    """
    avg_reward = test_summary["test_averages"]["Reward"]
    assert avg_reward >= threshold, (
        f"Learning failed: avg reward {avg_reward:.3f} < threshold {threshold:.3f}"
    )
    return True


def calculate_optimal_steps(agent_positions: list, food_positions: list) -> list:
    """Calculate optimal (beeline) steps from agent to food for each spawn.

    Stub for future implementation.

    Args:
        agent_positions: List of (x, y) tuples for agent spawn positions
        food_positions: List of (x, y) tuples for food spawn positions

    Returns:
        List of optimal step counts for each spawn
    """
    raise NotImplementedError("Optimal steps calculation not yet implemented")