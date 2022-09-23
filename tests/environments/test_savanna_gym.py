import pytest
import numpy as np
import numpy.testing as npt
from gym.spaces import Discrete


from aintelope.aintelope.environments.savanna_gym import SavannaGymEnv



def test_max_cycles():
    # currently the environment does not accept parameters like max_cycles
    # max_cycles_test(sut.env)
    pass


def test_render():
    # TODO: close method not implemented
    # render_test(sut.env)
    pass


def test_performance_benchmark():
    # TODO: probably gym has a test for this. research.
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
    # TODO: make this test appropriate to the Gym single agent env
    # env = sut.env()
    # assert len(env.possible_agents) == 1
    # env.reset()

    # agent = env.possible_agents[0]
    # for _ in range(sut.NUM_ITERS):
    #     action = {agent: env.action_space(agent).sample()}
    #     _, _, dones, _ = env.step(action)

    # assert dones[agent]
    # with pytest.raises(ValueError):
    #     action = {agent: env.action_space(agent).sample()}
    #     env.step(action)


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
