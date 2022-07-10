import numpy as np
import pytest
from pettingzoo.test import api_test, seed_test

from aintelope.environments import savanna as sut


def test_pettingzoo_api():
    api_test(sut.env(), num_cycles=1000)


def test_seed():
    seed_test(sut.env, num_cycles=10, test_kept_state=True)


def test_max_cycles():
    # TODO
    pass


def test_render():
    # TODO
    pass


def test_performance_benchmark():
    # TODO
    pass


def test_save_observation():
    # TODO
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


def test_agents():
    env = sut.env()
    with pytest.raises(AssertionError):
        env.state()
    assert len(env.possible_agents) == sut.AMOUNT_AGENTS

    env.reset()
    assert len(env.unwrapped.state) == sut.AMOUNT_AGENTS
    assert isinstance(env.unwrapped.state, dict)
    assert all(
        isinstance(agent_state, np.ndarray)
        for agent_state in env.unwrapped.state.values()
    )
    assert all(
        agent_state.shape == (2,)
        for agent_state in env.unwrapped.state.values()
    )


def test_grass_patches():
    env = sut.env()
    with pytest.raises(AttributeError):
        env.grass_patches
        env.unwrapped.grass_patches

    env.reset()
    assert len(env.unwrapped.grass_patches) == sut.AMOUNT_GRASS_PATCHES
    assert isinstance(env.unwrapped.grass_patches, np.ndarray)
    assert env.unwrapped.grass_patches.shape[1] == 2


def test_state():
    pass
