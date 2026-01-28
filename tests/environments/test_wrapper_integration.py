# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""
Integration tests for savanna_safetygrid wrapper contracts.

Tests verify:
- Reward scalarization (dict vs scalar output)
- Observation combination (tuple vs array output)
- Interoception vector length
"""

import numpy as np
import pytest

from aintelope.environments.savanna_safetygrid import (
    SavannaGridworldParallelEnv,
    SavannaGridworldSequentialEnv,
)


# =============================================================================
# REWARD SCALARIZATION
# =============================================================================

def test_reward_dict_parallel(base_test_config):
    """scalarize_rewards=False returns dict reward."""
    overrides = {"scalarize_rewards": False}
    env_params = dict(base_test_config.hparams.env_params)
    env_params.update(overrides)

    env = SavannaGridworldParallelEnv(env_params=env_params)
    obs, _ = env.reset()
    agent = next(iter(env.agents))
    actions = {agent: env.action_space(agent).sample()}
    _, rewards, _, _, _ = env.step(actions)

    assert isinstance(rewards[agent], dict)


def test_reward_dict_sequential(base_test_config):
    """scalarize_rewards=False returns dict reward."""
    overrides = {"scalarize_rewards": False}
    env_params = dict(base_test_config.hparams.env_params)
    env_params.update(overrides)

    env = SavannaGridworldSequentialEnv(env_params=env_params)
    env.reset()
    agent = env.agent_selection
    action = env.action_space(agent).sample()
    env.step(action)
    _, reward, _, _, _ = env.last()

    assert isinstance(reward, dict)


def test_reward_scalar_parallel(base_test_config):
    """scalarize_rewards=True returns numeric reward."""
    overrides = {"scalarize_rewards": True}
    env_params = dict(base_test_config.hparams.env_params)
    env_params.update(overrides)

    env = SavannaGridworldParallelEnv(env_params=env_params)
    obs, _ = env.reset()
    agent = next(iter(env.agents))
    actions = {agent: env.action_space(agent).sample()}
    _, rewards, _, _, _ = env.step(actions)

    assert isinstance(rewards[agent], (int, float, np.number))


def test_reward_scalar_sequential(base_test_config):
    """scalarize_rewards=True returns numeric reward."""
    overrides = {"scalarize_rewards": True}
    env_params = dict(base_test_config.hparams.env_params)
    env_params.update(overrides)

    env = SavannaGridworldSequentialEnv(env_params=env_params)
    env.reset()
    agent = env.agent_selection
    action = env.action_space(agent).sample()
    env.step(action)
    _, reward, _, _, _ = env.last()

    assert isinstance(reward, (int, float, np.number))


# =============================================================================
# OBSERVATION COMBINATION
# =============================================================================

def test_obs_tuple_parallel(base_test_config):
    """combine_interoception_and_vision=False returns (vision, interoception) tuple."""
    overrides = {"combine_interoception_and_vision": False}
    env_params = dict(base_test_config.hparams.env_params)
    env_params.update(overrides)

    env = SavannaGridworldParallelEnv(env_params=env_params)
    obs, _ = env.reset()
    agent = next(iter(env.agents))

    assert isinstance(obs[agent], tuple)
    assert len(obs[agent]) == 2
    assert isinstance(obs[agent][0], np.ndarray)
    assert isinstance(obs[agent][1], np.ndarray)


def test_obs_tuple_sequential(base_test_config):
    """combine_interoception_and_vision=False returns (vision, interoception) tuple."""
    overrides = {"combine_interoception_and_vision": False}
    env_params = dict(base_test_config.hparams.env_params)
    env_params.update(overrides)

    env = SavannaGridworldSequentialEnv(env_params=env_params)
    env.reset()
    agent = env.agent_selection
    obs = env.observe(agent)

    assert isinstance(obs, tuple)
    assert len(obs) == 2
    assert isinstance(obs[0], np.ndarray)
    assert isinstance(obs[1], np.ndarray)


def test_obs_array_parallel(base_test_config):
    """combine_interoception_and_vision=True returns single array."""
    overrides = {"combine_interoception_and_vision": True}
    env_params = dict(base_test_config.hparams.env_params)
    env_params.update(overrides)

    env = SavannaGridworldParallelEnv(env_params=env_params)
    obs, _ = env.reset()
    agent = next(iter(env.agents))

    assert isinstance(obs[agent], np.ndarray)
    assert not isinstance(obs[agent], tuple)


def test_obs_array_sequential(base_test_config):
    """combine_interoception_and_vision=True returns single array."""
    overrides = {"combine_interoception_and_vision": True}
    env_params = dict(base_test_config.hparams.env_params)
    env_params.update(overrides)

    env = SavannaGridworldSequentialEnv(env_params=env_params)
    env.reset()
    agent = env.agent_selection
    obs = env.observe(agent)

    assert isinstance(obs, np.ndarray)
    assert not isinstance(obs, tuple)


# =============================================================================
# INTEROCEPTION LENGTH
# =============================================================================

def test_interoception_length(base_test_config):
    """
    Interoception vector has expected length.
    
    TODO: Currently checks trivial case (length > 0). When config properly
    defines interoception modalities list, this should verify:
        len(interoception) == len(config.interoception_modalities) 
    """
    overrides = {"combine_interoception_and_vision": False}
    env_params = dict(base_test_config.hparams.env_params)
    env_params.update(overrides)

    env = SavannaGridworldParallelEnv(env_params=env_params)
    obs, _ = env.reset()
    agent = next(iter(env.agents))
    interoception = obs[agent][1]

    assert len(interoception) > 0