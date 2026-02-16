# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Repository:
# https://github.com/biological-alignment-benchmarks/biological-alignment-gridworlds-benchmarks

import os

import numpy as np
import pytest
from omegaconf import OmegaConf

from aintelope.environments import savanna_safetygrid as safetygrid
from pettingzoo.test.parallel_test import parallel_api_test
from pettingzoo.test.seed_test import parallel_seed_test
from gymnasium.spaces import Discrete


@pytest.mark.parametrize("execution_number", range(1))
def test_parallel_api(execution_number, base_env_cfg):
    """Smoke test: PettingZoo parallel API compliance."""
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
    env = safetygrid.SavannaGridworldParallelEnv(cfg=cfg)
    env.seed(execution_number)
    parallel_api_test(env, num_cycles=10)


@pytest.mark.parametrize("execution_number", range(1))
def test_parallel_seed(execution_number, base_env_cfg):
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

    def get_env_instance() -> safetygrid.SavannaGridworldParallelEnv:
        return safetygrid.SavannaGridworldParallelEnv(cfg=cfg)

    try:
        parallel_seed_test(get_env_instance, num_cycles=10)
    except TypeError:
        parallel_seed_test(get_env_instance)


@pytest.mark.parametrize("execution_number", range(1))
def test_parallel_step_result(execution_number, base_env_cfg):
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
    env = safetygrid.SavannaGridworldParallelEnv(cfg=cfg)
    num_agents = len(env.possible_agents)
    assert num_agents, f"expected 1 agent, got: {num_agents}"
    env.reset()

    agent = env.possible_agents[0]
    action = {agent: env.action_space(agent).sample()}

    observations, rewards, terminateds, truncateds, infos = env.step(action)
    dones = {
        key: terminated or truncateds[key] for (key, terminated) in terminateds.items()
    }

    assert not dones[agent]
    assert isinstance(observations, dict), "observations is not a dict"
    assert isinstance(
        observations[agent][0], np.ndarray
    ), "observations[0] is not an array"
    assert isinstance(
        observations[agent][1], np.ndarray
    ), "observations[1] is not an array"
    assert isinstance(rewards, dict), "rewards is not a dict"
    assert isinstance(rewards[agent], dict), "reward of agent is not a dict"


@pytest.mark.parametrize("execution_number", range(1))
def test_parallel_done_step(execution_number, base_env_cfg):
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
    env = safetygrid.SavannaGridworldParallelEnv(cfg=cfg)
    assert len(env.possible_agents) == 1
    env.reset()

    agent = env.possible_agents[0]
    for _ in range(env.metadata["num_iters"]):
        action = {agent: env.action_space(agent).sample()}
        _, _, terminateds, truncateds, _ = env.step(action)
        dones = {
            key: terminated or truncateds[key]
            for (key, terminated) in terminateds.items()
        }

    assert dones[agent]
    with pytest.raises(ValueError):
        action = {agent: env.action_space(agent).sample()}
        env.step(action)


def test_parallel_agents(base_env_cfg):
    """Agent registry is well-formed."""
    env = safetygrid.SavannaGridworldParallelEnv(cfg=base_env_cfg)

    assert isinstance(env.possible_agents, list)
    assert isinstance(env.unwrapped.agent_name_mapping, dict)
    assert all(
        agent_name in env.unwrapped.agent_name_mapping
        for agent_name in env.possible_agents
    )


def test_parallel_action_spaces(base_env_cfg):
    """Action spaces are Discrete with expected size."""
    env = safetygrid.SavannaGridworldParallelEnv(cfg=base_env_cfg)

    for agent in env.possible_agents:
        assert isinstance(env.action_space(agent), Discrete)
        assert env.action_space(agent).n == 5  # includes no-op


if __name__ == "__main__" and os.name == "nt":
    pytest.main([__file__])
