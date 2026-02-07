"""Integration test: agent completes full pipeline without errors."""

import os
import pytest
import pandas as pd
from omegaconf import OmegaConf

from aintelope.analytics.analytics import (
    assert_learning_threshold,
    assert_learning_improvement,
)
from aintelope.analytics.recording import read_events
from aintelope.__main__ import run
from tests.conftest import as_pipeline


def test_agent_completes_pipeline(learning_config):
    """Agent runs full train + test cycle without errors."""
    run(as_pipeline(learning_config))


def test_agent_learns(base_test_config):
    """Learning agent achieves reward threshold on simple gridworld."""

    # Minimal learning scenario overrides
    learning_overrides = {
        "agent_class": "sb3_ppo_agent",
        "num_episodes": 500,
        "test_episodes": 10,
        "show_plot": False,
        "do_not_enforce_checkpoint_file_existence_during_test": True,
        "env_layout_seed_repeat_sequence_length": 1,
        "model_params": {
            "num_conv_layers": 0,  # use MLP, not CNN
            "learning_rate": 0.001,
            "ppo_n_steps": 32,
        },
        "env_params": {
            "map_max": 4,
            "num_iters": 10,
            "combine_interoception_and_vision": True,
        },
    }
    cfg = OmegaConf.merge(base_test_config, OmegaConf.create(learning_overrides))

    result = run(as_pipeline(cfg))

    # Get config and summary for first experiment
    exp_cfg = result["configs"][0]
    summary = result["summaries"][0]

    # Build paths to events file
    experiment_dir = exp_cfg.experiment_dir
    events_fname = exp_cfg.events_fname

    # Read events and filter to training episodes
    events = read_events(experiment_dir, events_fname)
    events_combined = pd.concat(events, ignore_index=True)
    train_events = events_combined[~events_combined["IsTest"]]

    """ 
    # ========================================================================
    # DEBUG BLOCK - Remove this section after debugging
    # ========================================================================
    print("\n" + "="*60)
    print("DEBUG: Config Values (from exp_cfg)")
    print("="*60)
    print(f"num_episodes: {exp_cfg.hparams.num_episodes}")
    print(f"test_episodes: {exp_cfg.hparams.test_episodes}")
    print(f"num_iters: {exp_cfg.hparams.env_params.num_iters}")
    print(f"ppo_n_steps: {exp_cfg.hparams.model_params.get('ppo_n_steps', 'NOT SET')}")
    print(f"learning_rate: {exp_cfg.hparams.model_params.get('learning_rate', 'NOT SET')}")
    expected_total = exp_cfg.hparams.num_episodes * exp_cfg.hparams.env_params.num_iters
    print(f"Expected total_timesteps: {expected_total}")
    print(f"Expected episodes (at {exp_cfg.hparams.env_params.num_iters} steps/ep): {exp_cfg.hparams.num_episodes}")
    print("="*60)
    
    print("\n" + "="*60)
    print("DEBUG: Episode Analysis")
    print("="*60)
    print(f"Total episodes in events: {events_combined['Episode'].nunique()}")
    print(f"Train episodes (IsTest=False): {train_events['Episode'].nunique()}")
    print(f"Test episodes (IsTest=True): {events_combined[events_combined['IsTest']]['Episode'].nunique()}")
    
    train_episode_rewards = train_events.groupby("Episode")["Reward"].sum().sort_index()
    print(f"\nTrain episode rewards (total {len(train_episode_rewards)} episodes):")
    print("First 10 episodes:")
    print(train_episode_rewards.head(10))
    print("\nLast 10 episodes:")
    print(train_episode_rewards.tail(10))
    
    n_episodes = len(train_episode_rewards)
    window = max(1, int(n_episodes * 0.15))
    start_avg = train_episode_rewards.iloc[:window].mean()
    end_avg = train_episode_rewards.iloc[-window:].mean()
    print(f"\nWindow size (15%): {window} episodes")
    print(f"Start average (first {window}): {start_avg:.3f}")
    print(f"End average (last {window}): {end_avg:.3f}")
    print(f"Ratio: {end_avg/start_avg if start_avg > 0 else 'N/A'}")
    print("="*60 + "\n")
    
    exit()  # Remove this line after debugging
    # ========================================================================
    # END DEBUG BLOCK
    # ========================================================================
    """
    print("\n" + "=" * 60)
    print("\n" + "=" * 60)
    print("\n" + "=" * 60)
    print("\n" + "=" * 60)
    test_events = events_combined[events_combined["IsTest"]]

    test_episode_rewards = test_events.groupby("Episode")["Reward"].sum().sort_index()
    print("\n" + "=" * 60)
    print("DEBUG: Test Episode Analysis")
    print("=" * 60)
    print(f"Test episodes: {len(test_episode_rewards)}")
    print(test_episode_rewards)
    print(f"Average test reward: {test_episode_rewards.mean():.3f}")
    print("=" * 60 + "\n")
    print("=" * 60 + "\n")
    print("=" * 60 + "\n")
    print("=" * 60 + "\n")

    print("\n" + "=" * 60)
    print("DEBUG: Test Episode Step Counts")
    print("=" * 60)
    test_steps_per_episode = test_events.groupby("Episode").size()
    print(test_steps_per_episode)
    print(f"Average steps per test episode: {test_steps_per_episode.mean():.1f}")

    # Check action distribution in test vs train
    print("\nAction distribution in test:")
    print(test_events["Action"].value_counts().sort_index())
    print("\nAction distribution in last 50 training episodes:")
    late_train = train_events[train_events["Episode"] >= 450]
    print(late_train["Action"].value_counts().sort_index())
    print("=" * 60 + "\n")

    # Assert learning improvement from early to late training
    assert_learning_improvement(train_events)


if __name__ == "__main__":
    pytest.main([__file__])
