# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Unit tests for GridworldGymWrapper and flatten_obs."""

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
from aintelope.environments.gridworld_gym_wrapper import (
    GridworldGymWrapper,
    flatten_obs,
)


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
    return GridworldGymWrapper(env, "agent_0"), env


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
        assert flatten_obs(obs).shape == (12, 5, 5)

    def test_interoception_broadcast(self):
        obs = {
            "vision": np.zeros((3, 4, 4), dtype=np.float32),
            "interoception": np.array([0.5, 1.0], dtype=np.float32),
        }
        out = flatten_obs(obs)
        assert np.all(out[3] == 0.5)
        assert np.all(out[4] == 1.0)

    def test_vision_channels_preserved(self):
        vision = np.random.rand(3, 4, 4).astype(np.float32)
        obs = {"vision": vision, "interoception": np.zeros(2, dtype=np.float32)}
        np.testing.assert_array_equal(flatten_obs(obs)[:3], vision)


# ── Spaces ─────────────────────────────────────────────────────────────────────


class TestSpaces:
    def test_observation_space_is_box(self):
        wrapper, _ = _make()
        assert isinstance(wrapper.observation_space, Box)

    def test_observation_space_shape_matches_flatten_obs(self):
        wrapper, env = _make()
        obs_dict, _ = env.reset()
        assert wrapper.observation_space.shape == flatten_obs(obs_dict["agent_0"]).shape

    def test_action_space_is_discrete(self):
        wrapper, _ = _make()
        assert isinstance(wrapper.action_space, Discrete)

    def test_action_space_n_matches_manifesto(self):
        wrapper, env = _make()
        assert wrapper.action_space.n == len(env.manifesto["action_space"])


# ── Reset ──────────────────────────────────────────────────────────────────────


class TestReset:
    def test_returns_obs_and_info(self):
        wrapper, _ = _make()
        obs, info = wrapper.reset()
        assert isinstance(obs, np.ndarray)
        assert isinstance(info, dict)

    def test_obs_shape_correct(self):
        wrapper, _ = _make()
        obs, _ = wrapper.reset()
        assert obs.shape == wrapper.observation_space.shape


# ── Step ───────────────────────────────────────────────────────────────────────


class TestStep:
    def test_returns_five_tuple(self):
        wrapper, _ = _make()
        assert len(wrapper.step(0)) == 5

    def test_obs_shape_correct_after_step(self):
        wrapper, _ = _make()
        obs, _, _, _, _ = wrapper.step(0)
        assert obs.shape == wrapper.observation_space.shape

    def test_reward_is_food_interoception(self):
        wrapper, env = _make()
        env._board[1:-1, 1:-1] = FLOOR
        _place(env, (5, 5), _N)
        env._board[4, 5] = FOOD
        fwd = env.manifesto["action_space"].index("forward")
        _, reward, _, _, _ = wrapper.step(fwd)
        assert reward == 1.0

    def test_reward_zero_on_empty_step(self):
        wrapper, env = _make()
        env._board[1:-1, 1:-1] = FLOOR
        _place(env, (5, 5), _N)
        fwd = env.manifesto["action_space"].index("forward")
        _, reward, _, _, _ = wrapper.step(fwd)
        assert reward == 0.0

    def test_truncated_always_false(self):
        wrapper, _ = _make()
        _, _, _, truncated, _ = wrapper.step(0)
        assert truncated is False

    def test_terminated_false_on_no_done(self):
        wrapper, _ = _make()
        _, _, terminated, _, _ = wrapper.step(0)
        assert terminated is False

    def test_terminated_true_on_food_done(self):
        wrapper, env = _make(termination="food")
        env._board[1:-1, 1:-1] = FLOOR
        _place(env, (5, 5), _N)
        env._board[4, 5] = FOOD
        fwd = env.manifesto["action_space"].index("forward")
        _, _, terminated, _, _ = wrapper.step(fwd)
        assert terminated is True
