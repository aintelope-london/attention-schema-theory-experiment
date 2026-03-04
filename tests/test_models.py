"""Unit tests for Model connectome — init and activation flow."""

import os
import pytest
from omegaconf import OmegaConf
from aintelope.agents.model.model import Model
from aintelope.environments import get_env_class


@pytest.fixture
def model_free_cfg():
    cfg = OmegaConf.load(os.path.join("aintelope", "config", "default_config.yaml"))
    return OmegaConf.merge(
        cfg,
        {
            "env_params": {
                "map_max": 5,
                "combine_interoception_and_vision": False,
            },
            "agent_params": {
                "roi_mode": None,
                "agent_0": {
                    "agent_class": "main_agent",
                    "architecture": {
                        "action": {"type": "DQN", "inputs": ["q_net"]},
                        "reward": {
                            "type": "RewardInference",
                            "inputs": ["observation"],
                        },
                        "q_net": {"type": "DQN-NN", "inputs": ["observation"]},
                    },
                },
            },
        },
    )


@pytest.fixture
def model_free_env(model_free_cfg):
    env_cls = get_env_class(model_free_cfg.env_params.env)
    e = env_cls(model_free_cfg)
    e.reset()
    return e


@pytest.fixture
def model_based_cfg():
    cfg = OmegaConf.load(os.path.join("aintelope", "config", "default_config.yaml"))
    return OmegaConf.merge(
        cfg,
        {
            "env_params": {
                "map_max": 5,
                "combine_interoception_and_vision": False,
            },
            "agent_params": {
                "roi_mode": None,
                "agent_0": {
                    "agent_class": "main_agent",
                    "architecture": {
                        "action": {
                            "type": "ModelBased",
                            "inputs": ["dynamic", "value"],
                        },
                        "reward": {
                            "type": "RewardInference",
                            "inputs": ["observation"],
                        },
                        "dynamic": {"type": "NextState-NN", "inputs": ["observation"]},
                        "value": {"type": "StateValue-NN", "inputs": ["observation"]},
                    },
                },
            },
        },
    )


@pytest.fixture
def model_based_env(model_based_cfg):
    env_cls = get_env_class(model_based_cfg.env_params.env)
    e = env_cls(model_based_cfg)
    e.reset()
    return e


class TestModelFree:
    """DQN architecture — init and step cycle."""

    def test_init(self, model_free_env, model_free_cfg):
        model = Model("agent_0", model_free_env.manifesto, model_free_cfg)
        assert "action" in model.components
        assert "reward" in model.components
        assert "q_net" in model.components

    def test_step_cycle(self, model_free_env, model_free_cfg):
        model = Model("agent_0", model_free_env.manifesto, model_free_cfg)
        observations, _ = model_free_env.reset()
        obs = observations["agent_0"]

        for _ in range(3):
            action = model.get_action(obs)
            assert isinstance(action, int)

            observations, _, _, _, _ = model_free_env.step_parallel({"agent_0": action})
            next_obs = observations["agent_0"]
            model.update(next_obs)
            assert len(model.activations) == 0
            obs = next_obs


class TestModelBased:
    """MCTS architecture — init and step cycle."""

    def test_init(self, model_based_env, model_based_cfg):
        model = Model("agent_0", model_based_env.manifesto, model_based_cfg)
        assert "action" in model.components
        assert "reward" in model.components
        assert "dynamic" in model.components
        assert "value" in model.components

    def test_step_cycle(self, model_based_env, model_based_cfg):
        model = Model("agent_0", model_based_env.manifesto, model_based_cfg)
        observations, _ = model_based_env.reset()
        obs = observations["agent_0"]

        for _ in range(3):
            action = model.get_action(obs)
            assert isinstance(action, int)

            observations, _, _, _, _ = model_based_env.step_parallel(
                {"agent_0": action}
            )
            next_obs = observations["agent_0"]
            model.update(next_obs)
            assert len(model.activations) == 0
            obs = next_obs
