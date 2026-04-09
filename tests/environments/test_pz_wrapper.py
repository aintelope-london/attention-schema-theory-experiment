"""Unit tests for GridworldPZWrapper and flatten_obs."""

import numpy as np
import pytest
from gymnasium.spaces import Box, Discrete
from omegaconf import OmegaConf

from aintelope.environments.gridworld import (
    FLOOR,
    FOOD,
    _N,
    _N_BASE,
    GridworldEnv,
)
from aintelope.environments.gridworld_pz_wrapper import GridworldPZWrapper, flatten_obs


# ── Helpers ────────────────────────────────────────────────────────────────────


def _cfg(map_size=9, termination=None):
    return OmegaConf.create(
        {
            "run": {"seed": 0},
            "env_params": {
                "map_size": map_size,
                "objects": {"food": {"count": 1}, "predator": {"count": 0}},
                "actions": ["forward", "left", "right", "backward"],
                "observation_radius": 2,
                "observation_format": "boolean_cube",
                "termination": termination,
            },
            "agent_params": {"agents": {"agent_0": {"agent_class": "main_agent"}}},
        }
    )


def _make(termination=None):
    env = GridworldEnv(_cfg(termination=termination))
    env.reset()
    return GridworldPZWrapper(env), env


def _fwd(wrapper):
    fwd = wrapper._env.manifesto["action_space"].index("forward")
    return {"agent_0": fwd}


def _place(env, pos, facing=_N):
    old = env._positions["agent_0"]
    env._board[old] = FLOOR
    env._positions["agent_0"] = pos
    env._board[pos] = _N_BASE + env.agents.index("agent_0")
    env._facing["agent_0"] = facing


# ── flatten_obs ────────────────────────────────────────────────────────────────


class TestFlattenObs:
    def test_output_shape(self):
        obs = {
            "vision": np.zeros((10, 5, 5), dtype=np.float32),
            "interoception": np.zeros(2, dtype=np.float32),
        }
        out = flatten_obs(obs)
        assert out.shape == (12, 5, 5)

    def test_interoception_broadcast(self):
        obs = {
            "vision": np.zeros((3, 4, 4), dtype=np.float32),
            "interoception": np.array([0.5, 1.0], dtype=np.float32),
        }
        out = flatten_obs(obs)
        # interoception[0] = 0.5 should fill channel 3 entirely
        assert np.all(out[3] == 0.5)
        # interoception[1] = 1.0 should fill channel 4 entirely
        assert np.all(out[4] == 1.0)

    def test_vision_channels_preserved(self):
        vision = np.random.rand(3, 4, 4).astype(np.float32)
        obs = {
            "vision": vision,
            "interoception": np.zeros(2, dtype=np.float32),
        }
        out = flatten_obs(obs)
        np.testing.assert_array_equal(out[:3], vision)


# ── Spaces ─────────────────────────────────────────────────────────────────────


class TestSpaces:
    def test_observation_space_is_box(self):
        wrapper, _ = _make()
        assert isinstance(wrapper.observation_space("agent_0"), Box)

    def test_observation_space_shape_matches_flatten_obs(self):
        wrapper, env = _make()
        obs_dict, _ = env.reset()
        flat = flatten_obs(obs_dict["agent_0"])
        assert wrapper.observation_space("agent_0").shape == flat.shape

    def test_action_space_is_discrete(self):
        wrapper, _ = _make()
        assert isinstance(wrapper.action_space("agent_0"), Discrete)

    def test_action_space_n_matches_manifesto(self):
        wrapper, env = _make()
        n = len(env.manifesto["action_space"])
        assert wrapper.action_space("agent_0").n == n


# ── Agents ─────────────────────────────────────────────────────────────────────


class TestAgents:
    def test_possible_agents_matches_env(self):
        wrapper, env = _make()
        assert wrapper.possible_agents == env.agents

    def test_num_agents(self):
        wrapper, _ = _make()
        assert wrapper.num_agents == 1


# ── Reset ──────────────────────────────────────────────────────────────────────


class TestReset:
    def test_returns_obs_and_infos(self):
        wrapper, _ = _make()
        result = wrapper.reset()
        assert len(result) == 2

    def test_obs_shape_correct(self):
        wrapper, _ = _make()
        obs, _ = wrapper.reset()
        expected = wrapper.observation_space("agent_0").shape
        assert obs["agent_0"].shape == expected

    def test_infos_is_dict(self):
        wrapper, _ = _make()
        _, infos = wrapper.reset()
        assert isinstance(infos, dict)
        assert "agent_0" in infos

    def test_agents_repopulated_on_reset(self):
        wrapper, _ = _make(termination="food")
        env = wrapper._env
        env._board[1:-1, 1:-1] = FLOOR
        _place(env, (5, 5), _N)
        env._board[4, 5] = FOOD
        wrapper.step(_fwd(wrapper))  # triggers done, empties agents
        assert wrapper.agents == []
        wrapper.reset()
        assert wrapper.agents == ["agent_0"]


# ── Step ───────────────────────────────────────────────────────────────────────


class TestStep:
    def test_returns_five_tuple(self):
        wrapper, _ = _make()
        result = wrapper.step(_fwd(wrapper))
        assert len(result) == 5

    def test_obs_shape_correct_after_step(self):
        wrapper, _ = _make()
        obs, _, _, _, _ = wrapper.step(_fwd(wrapper))
        assert obs["agent_0"].shape == wrapper.observation_space("agent_0").shape

    def test_reward_is_food_interoception(self):
        wrapper, env = _make()
        env._board[1:-1, 1:-1] = FLOOR
        _place(env, (5, 5), _N)
        env._board[4, 5] = FOOD
        _, rewards, _, _, _ = wrapper.step(_fwd(wrapper))
        assert rewards["agent_0"] == 1.0

    def test_reward_zero_on_empty_step(self):
        wrapper, env = _make()
        env._board[1:-1, 1:-1] = FLOOR
        _place(env, (5, 5), _N)
        _, rewards, _, _, _ = wrapper.step(_fwd(wrapper))
        assert rewards["agent_0"] == 0.0

    def test_truncated_always_false(self):
        wrapper, _ = _make()
        _, _, _, truncateds, _ = wrapper.step(_fwd(wrapper))
        assert truncateds["agent_0"] is False

    def test_terminated_false_on_no_done(self):
        wrapper, _ = _make()
        _, _, terminateds, _, _ = wrapper.step(_fwd(wrapper))
        assert terminateds["agent_0"] is False

    def test_terminated_true_on_food_done(self):
        wrapper, env = _make(termination="food")
        env._board[1:-1, 1:-1] = FLOOR
        _place(env, (5, 5), _N)
        env._board[4, 5] = FOOD
        _, _, terminateds, _, _ = wrapper.step(_fwd(wrapper))
        assert terminateds["agent_0"] is True

    def test_agents_shrinks_when_done(self):
        wrapper, env = _make(termination="food")
        env._board[1:-1, 1:-1] = FLOOR
        _place(env, (5, 5), _N)
        env._board[4, 5] = FOOD
        wrapper.step(_fwd(wrapper))
        assert wrapper.agents == []
