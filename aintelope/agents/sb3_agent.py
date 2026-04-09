# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""SB3 baseline agent.

Single class for all SB3 algorithms. Algorithm is config-driven
(cfg.agent_params.algorithm: ppo | dqn | a2c).

Remove by: deleting this file and gridworld_pz_wrapper.py,
           one register_agent_class line in agents/__init__.py,
           one if-branch in experiment.py.
"""

import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig

import stable_baselines3
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from zoo_to_gym_multiagent_adapter.singleagent_zoo_to_gym_adapter import (
    SingleAgentZooToGymAdapter,
)

from aintelope.agents.abstract_agent import AbstractAgent
from aintelope.environments.gridworld_pz_wrapper import GridworldPZWrapper, flatten_obs

_ALGORITHMS = {"ppo": PPO, "dqn": DQN, "a2c": A2C}


# ── CNN feature extractor ──────────────────────────────────────────────────────


class _CustomCNN(BaseFeaturesExtractor):
    """Flexible CNN for SB3 policies. num_conv_layers=0 → flatten only."""

    def __init__(self, observation_space, features_dim=256, num_conv_layers=2):
        super().__init__(observation_space, features_dim)
        C, H, W = observation_space.shape
        layers = []
        in_ch = C
        for i in range(num_conv_layers):
            out_ch = 32 if i == 0 else 64
            stride = 1 if i == 0 else 2
            layers += [nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1), nn.ReLU()]
            in_ch = out_ch
        layers.append(nn.Flatten())
        self.cnn = nn.Sequential(*layers)
        with torch.no_grad():
            n_flat = self.cnn(torch.zeros(1, C, H, W)).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flat, features_dim), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(x))


# ── Helpers ────────────────────────────────────────────────────────────────────


def _is_json_serializable(_):
    return False


def _build_model(gym_env, cfg):
    algo = cfg.agent_params.get("algorithm", "ppo").lower()
    n_conv = cfg.agent_params.get("num_conv_layers", 0)
    policy = "CnnPolicy" if n_conv > 0 else "MlpPolicy"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy_kwargs = {"normalize_images": False}
    if n_conv > 0:
        policy_kwargs.update(
            features_extractor_class=_CustomCNN,
            features_extractor_kwargs={"features_dim": 256, "num_conv_layers": n_conv},
        )

    cls = _ALGORITHMS[algo]
    kwargs = dict(
        policy=policy,
        env=gym_env,
        verbose=0,
        device=device,
        policy_kwargs=policy_kwargs,
        learning_rate=cfg.agent_params.get("learning_rate", 3e-4),
    )
    if algo == "dqn":
        kwargs.update(
            learning_rate=cfg.agent_params.get("learning_rate", 1e-4),
            buffer_size=cfg.agent_params.get("replay_buffer_size", 30000),
            batch_size=cfg.agent_params.get("batch_size", 32),
            gamma=cfg.agent_params.get("gamma", 0.99),
            optimize_memory_usage=True,
            replay_buffer_kwargs={"handle_timeout_termination": False},
        )
    elif algo == "ppo":
        kwargs["n_steps"] = cfg.agent_params.get("ppo_n_steps", 64)

    return cls(**kwargs)


# ── Agent ─────────────────────────────────────────────────────────────────────


class SB3Agent(AbstractAgent):
    """SB3 baseline. Implements AbstractAgent; algorithm is config-driven.

    Training: static method training() is called directly from experiment.py
              under the documented special-permission branch.
    Test:     runs through the canonical experiment loop via get_action().
    """

    def __init__(
        self,
        agent_id: str,
        env=None,
        cfg: DictConfig = None,
        checkpoint: Optional[str] = None,
        **kwargs,
    ):
        self.id = agent_id
        self.cfg = cfg
        self.last_action = None
        self.done = False

        stable_baselines3.common.save_util.is_json_serializable = _is_json_serializable

        wrapper = GridworldPZWrapper(env)
        gym_env = SingleAgentZooToGymAdapter(wrapper, agent_id)
        self.model = _build_model(gym_env, cfg)

        if checkpoint:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.set_parameters(
                torch.load(checkpoint, map_location=device), device=device
            )

    # ── AbstractAgent contract ─────────────────────────────────────────────────

    def reset(self, state, **kwargs) -> None:
        self.done = False
        self.last_action = None

    def get_action(self, observation=None, **kwargs) -> dict:
        obs = flatten_obs(observation)
        action, _ = self.model.predict(obs, deterministic=True)
        self.last_action = int(np.asarray(action).item())
        return {"action": self.last_action}

    def update(self, observation=None, **kwargs) -> dict:
        return {}

    def save_model(self, path, **kwargs) -> None:
        pass  # training() saves its own checkpoint; test-mode model is stateless.

    # ── SB3 training (special permission — documented in DOCUMENTATION.md) ─────

    @staticmethod
    def training(env, num_total_steps: int, cfg: DictConfig, i_trial: int) -> None:
        """Runs SB3's internal training loop. Called once per train block.

        Owns the wrapper, gym adapter, model lifecycle, and checkpoint saving.
        Nothing outside this method needs to know how SB3 training works.
        """
        from aintelope.agents.model.dl_utils import checkpoint_path

        stable_baselines3.common.save_util.is_json_serializable = _is_json_serializable
        agent_id = next(iter(cfg.agent_params.agents))
        wrapper = GridworldPZWrapper(env)
        gym_env = SingleAgentZooToGymAdapter(wrapper, agent_id)
        model = _build_model(gym_env, cfg)
        model.learn(total_timesteps=num_total_steps)
        path = checkpoint_path(cfg.run.outputs_dir, agent_id, i_trial)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(model.get_parameters(), path)
