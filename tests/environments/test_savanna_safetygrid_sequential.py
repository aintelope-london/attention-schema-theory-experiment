# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Repository:
# https://github.com/aintelope-london/attention-schema-theory-experiment

import os

import numpy as np
import pytest
from omegaconf import OmegaConf

from aintelope.environments import savanna_safetygrid as safetygrid
from pettingzoo.test import api_test
from pettingzoo.test.seed_test import seed_test
from gymnasium.spaces import Discrete


@pytest.mark.parametrize("execution_number", range(1))
def test_sequential_api(execution_number, base_env_cfg):
    """Smoke test: PettingZoo sequential API compliance."""
    cfg = OmegaConf.merge(
        base_env_cfg,
        {
            "env_params": {
                "num_iters": 500,
                "map_min": 0,
                "map_max": 100,
                "render_map_max": 100,
                "amount_agents": 1,
                "amount_grass_patches": 2,
                "amount_water_holes": 2,
                "scalarize_rewards": True,
            }
        },
    )
    env = safetygrid.SavannaGridworldSequentialEnv(cfg=cfg)
    env.seed(execution_number)
    api_test(env, num_cycles=10, verbose_progress=True)


@pytest.mark.parametrize("execution_number", range(1))
def test_sequential_seed(execution_number, base_env_cfg):
    """Reproducibility: same seed produces same trajectory."""
    cfg = OmegaConf.merge(
        base_env_cfg,
        {
            "env_params": {
                "override_infos": True,
                "seed": execution_number,
            }
        },
    )

    def get_env_instance() -> safetygrid.SavannaGridworldSequentialEnv:
        return safetygrid.SavannaGridworldSequentialEnv(cfg=cfg)

    try:
        seed_test(get_env_instance, num_cycles=10)
    except TypeError:
        seed_test(get_env_instance)


@pytest.mark.parametrize("execution_number", range(1))
def test_sequential_step_result(execution_number, base_env_cfg):
    """Step returns correct observation/reward structure."""
    cfg = OmegaConf.merge(
        base_env_cfg,
        {
            "env_params": {
                "num_iters": 2,
                "seed": execution_number,
            }
        },
    )
    env = safetygrid.SavannaGridworldSequentialEnv(cfg=cfg)
    num_agents = len(env.possible_agents)
    assert num_agents, f"expected 1 agent, got: {num_agents}"
    env.reset()

    agent = env.agent_selection
    action = env.action_space(agent).sample()

    env.step(action)
    observation, reward, terminated, truncated, info = env.last()
    done = terminated or truncated

    assert not done

    if not env._combine_interoception_and_vision:
        assert isinstance(observation[0], np.ndarray), "observation[0] is not an array"
        assert isinstance(observation[1], np.ndarray), "observation[1] is not an array"
    else:
        assert isinstance(observation, np.ndarray), "observation is not an array"

    assert isinstance(reward, dict), "reward of agent is not a dict"


@pytest.mark.parametrize("execution_number", range(1))
def test_sequential_done_step(execution_number, base_env_cfg):
    """Environment terminates after num_iters and rejects further steps."""
    cfg = OmegaConf.merge(
        base_env_cfg,
        {
            "env_params": {
                "amount_agents": 1,
                "seed": execution_number,
            }
        },
    )
    env = safetygrid.SavannaGridworldSequentialEnv(cfg=cfg)
    assert len(env.possible_agents) == 1
    env.reset()

    for _ in range(env.metadata["num_iters"]):
        agent = env.agent_selection
        action = env.action_space(agent).sample()
        env.step(action)
        terminated = env.terminations[agent]
        truncated = env.truncations[agent]
        done = terminated or truncated

    assert done
    with pytest.raises(ValueError):
        action = env.action_space(agent).sample()
        env.step(action)


def test_sequential_agents(base_env_cfg):
    """Agent registry is well-formed."""
    env = safetygrid.SavannaGridworldSequentialEnv(cfg=base_env_cfg)

    assert isinstance(env.possible_agents, list)
    assert isinstance(env.unwrapped.agent_name_mapping, dict)
    assert all(
        agent_name in env.unwrapped.agent_name_mapping
        for agent_name in env.possible_agents
    )


def test_sequential_action_spaces(base_env_cfg):
    """Action spaces are Discrete with expected size."""
    env = safetygrid.SavannaGridworldSequentialEnv(cfg=base_env_cfg)

    for agent in env.possible_agents:
        assert isinstance(env.action_space(agent), Discrete)
        assert env.action_space(agent).n == 5  # includes no-op


if __name__ == "__main__" and os.name == "nt":
    pytest.main([__file__])
