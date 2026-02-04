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

def assert_learning_improvement(
    events: pd.DataFrame,
    episode_fraction: float = 0.15,
    min_improvement_ratio: float = 1.3,
) -> None:
    """Check if agent improved from early to late episodes.
    
    Args:
        events: DataFrame with columns Episode, Reward, IsTest
        episode_fraction: Fraction of episodes to compare at start/end (0.15 = 15%)
        min_improvement_ratio: end_avg must be >= start_avg * this ratio
    
    Raises:
        AssertionError if improvement not detected
    """
    # Aggregate to per-episode reward totals
    episode_rewards = events.groupby("Episode")["Reward"].sum().sort_index()
    
    n_episodes = len(episode_rewards)
    window = max(1, int(n_episodes * episode_fraction))
    
    start_avg = episode_rewards.iloc[:window].mean()
    end_avg = episode_rewards.iloc[-window:].mean()
    
    if start_avg <= 0:
        assert end_avg > start_avg, (
            f"No improvement: start_avg={start_avg:.3f}, end_avg={end_avg:.3f}"
        )
    else:
        ratio = end_avg / start_avg
        assert ratio >= min_improvement_ratio, (
            f"Insufficient improvement: ratio={ratio:.2f}x < {min_improvement_ratio}x "
            f"(start={start_avg:.3f}, end={end_avg:.3f}, window={window} episodes)"
        )

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