import pytest
import numpy as np
import numpy.testing as npt
from gym.spaces import Discrete
from pettingzoo.test import (
    max_cycles_test,
    render_test,
    performance_benchmark,
)
from pettingzoo.test.parallel_test import parallel_api_test
from pettingzoo.test import api_test
from pettingzoo.test.seed_test import parallel_seed_test
from pettingzoo.utils import parallel_to_aec

from aintelope.aintelope.environments import savanna_zoo as sut
from aintelope.aintelope.environments.savanna_zoo import SavannaZooEnv


def test_pettingzoo_api_parallel():
    parallel_api_test(sut.env(), num_cycles=1000)
    
    
    
def test_pettingzoo_api_sequential():
    # TODO: refactor these values out to a test-params file
    env_params = {
        'num_iters': 500,  # duration of the game
        'map_min': 0,
        'map_max': 100,
        'render_map_max': 100,
        'amount_agents': 1,  # for now only one agent
        'amount_grass_patches': 2,
        'amount_water_holes': 2,
    }
    parallel_env = SavannaZooEnv(env_params=env_params)
    # TODO: Nathan was able to get the sequential-turn env to work, using this conversion, but not the parallel env. why??
    sequential_env = parallel_to_aec(parallel_env)
    api_test(sequential_env, num_cycles=10, verbose_progress=True)


def test_seed():
    parallel_seed_test(sut.env, num_cycles=10, test_kept_state=True)


def test_max_cycles():
    # currently the environment does not accept parameters like max_cycles
    # max_cycles_test(sut.env)
    pass


def test_render():
    # TODO: close method not implemented
    # render_test(sut.env)
    pass


def test_performance_benchmark():
    # will print only timing to stdout; not shown per default
    # performance_benchmark(sut.env())
    pass


def test_reward_agent():
    # single grass patch
    agent_pos = np.random.randint(sut.MAP_MIN, sut.MAP_MAX, 2)
    grass_patch = np.random.randint(sut.MAP_MIN, sut.MAP_MAX, 2)

    reward_single = sut.reward_agent(agent_pos, grass_patch)
    assert reward_single == 1 / (1 + sut.vec_distance(grass_patch, agent_pos))

    # multiple grass patches
    grass_patches = np.random.randint(sut.MAP_MIN, sut.MAP_MAX, size=(10, 2))
    reward_many = sut.reward_agent(agent_pos, grass_patches)
    grass_patch_closest = grass_patches[
        np.argmin(
            np.linalg.norm(np.subtract(grass_patches, agent_pos), axis=1)
        )
    ]
    assert reward_many == 1 / (
        1 + sut.vec_distance(grass_patch_closest, agent_pos)
    )


def test_move_agent():
    env = sut.env()
    env.reset()

    agent = env.possible_agents[0]
    agent_states = env.unwrapped.agent_states

    for _ in range(1000):
        prev_state = np.copy(agent_states[agent])
        action = env.action_space(agent).sample()
        agent_states[agent] = sut.move_agent(agent_states[agent], action)
        npt.assert_array_equal(
            np.clip(
                prev_state + sut.ACTION_MAP[action], sut.MAP_MIN, sut.MAP_MAX
            ),
            agent_states[agent],
        )
        assert sut.MAP_MIN <= agent_states[agent][0] <= sut.MAP_MAX
        assert sut.MAP_MIN <= agent_states[agent][1] <= sut.MAP_MAX
        assert agent_states[agent].dtype == sut.PositionFloat


def test_done_step():
    env = sut.env()
    assert len(env.possible_agents) == 1
    env.reset()

    agent = env.possible_agents[0]
    for _ in range(sut.NUM_ITERS):
        action = {agent: env.action_space(agent).sample()}
        _, _, dones, _ = env.step(action)

    assert dones[agent]
    with pytest.raises(ValueError):
        action = {agent: env.action_space(agent).sample()}
        env.step(action)


def test_agents():
    env = sut.env()

    assert len(env.possible_agents) == sut.AMOUNT_AGENTS
    assert isinstance(env.possible_agents, list)
    assert isinstance(env.unwrapped.agent_name_mapping, dict)
    assert all(
        agent_name in env.unwrapped.agent_name_mapping
        for agent_name in env.possible_agents
    )


def test_agent_states():
    env = sut.env()

    with pytest.raises(AttributeError):
        env.agent_states
    with pytest.raises(AttributeError):
        env.unwrapped.agent_states

    env.reset()
    assert isinstance(env.unwrapped.agent_states, dict)
    assert all(
        isinstance(agent_state, np.ndarray)
        for agent_state in env.unwrapped.agent_states.values()
    )
    assert all(
        agent_state.shape == (2,)
        for agent_state in env.unwrapped.agent_states.values()
    )


def test_action_spaces():
    env = sut.env()

    for agent in env.possible_agents:
        assert isinstance(env.action_space(agent), Discrete)
        assert env.action_space(agent).n == 4


def test_grass_patches():
    env = sut.env()

    with pytest.raises(AttributeError):
        env.grass_patches
    with pytest.raises(AttributeError):
        env.unwrapped.grass_patches

    env.reset()
    assert len(env.unwrapped.grass_patches) == sut.AMOUNT_GRASS_PATCHES
    assert isinstance(env.unwrapped.grass_patches, np.ndarray)
    assert env.unwrapped.grass_patches.shape[1] == 2


def test_observation_spaces():
    pass  # TODO
