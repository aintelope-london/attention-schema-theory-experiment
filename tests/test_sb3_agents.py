# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Unit tests for SB3Agent.

Covers: model construction, AbstractAgent contract methods,
action range validity, save_model no-op.
Does not test training() — that belongs in tests/learning/.
"""

import numpy as np
import pytest
from omegaconf import OmegaConf

from aintelope.environments.gridworld import GridworldEnv
from aintelope.agents.sb3_agent import SB3Agent


# ── Helpers ────────────────────────────────────────────────────────────────────


def _env_cfg(map_size=3):
    return OmegaConf.create(
        {
            "run": {"seed": 0},
            "env_params": {
                "map_size": map_size,
                "objects": {"food": {"count": 1}, "predator": {"count": 0}},
                "actions": ["forward", "left", "right", "backward"],
                "observation_radius": 2,
                "observation_format": "boolean_cube",
                "termination": "food",
            },
            "agent_params": {
                "agents": {"agent_0": {"agent_class": "sb3_agent"}},
                "algorithm": "ppo",
                "num_conv_layers": 0,
                "ppo_n_steps": 64,
                "learning_rate": 3e-4,
            },
        }
    )


def _make():
    cfg = _env_cfg()
    env = GridworldEnv(cfg)
    obs, _ = env.reset()
    agent = SB3Agent(agent_id="agent_0", env=env, cfg=cfg)
    return agent, obs["agent_0"]


# ── Init ───────────────────────────────────────────────────────────────────────


class TestInit:
    def test_model_created(self):
        agent, _ = _make()
        assert agent.model is not None

    def test_last_action_none_at_init(self):
        agent, _ = _make()
        assert agent.last_action is None

    def test_done_false_at_init(self):
        agent, _ = _make()
        assert agent.done is False


# ── Reset ──────────────────────────────────────────────────────────────────────


class TestReset:
    def test_clears_last_action(self):
        agent, obs = _make()
        agent.get_action(observation=obs)
        assert agent.last_action is not None
        agent.reset(obs)
        assert agent.last_action is None

    def test_clears_done(self):
        agent, obs = _make()
        agent.done = True
        agent.reset(obs)
        assert agent.done is False


# ── get_action ─────────────────────────────────────────────────────────────────


class TestGetAction:
    def test_returns_dict(self):
        agent, obs = _make()
        result = agent.get_action(observation=obs)
        assert isinstance(result, dict)

    def test_has_action_key(self):
        agent, obs = _make()
        result = agent.get_action(observation=obs)
        assert "action" in result

    def test_action_in_valid_range(self):
        agent, obs = _make()
        cfg = _env_cfg()
        n_actions = len(cfg.env_params.actions)
        for _ in range(10):
            result = agent.get_action(observation=obs)
            assert 0 <= result["action"] < n_actions

    def test_sets_last_action(self):
        agent, obs = _make()
        agent.get_action(observation=obs)
        assert agent.last_action is not None


# ── update ─────────────────────────────────────────────────────────────────────


class TestUpdate:
    def test_returns_dict(self):
        agent, obs = _make()
        result = agent.update(observation=obs)
        assert isinstance(result, dict)


# ── save_model ─────────────────────────────────────────────────────────────────


class TestSaveModel:
    def test_is_noop(self, tmp_path):
        agent, _ = _make()
        path = str(tmp_path / "model.pt")
        agent.save_model(path)  # must not raise
        import os

        assert not os.path.exists(path)
