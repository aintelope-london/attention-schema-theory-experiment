# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Gymnasium wrapper around GridworldEnv for SB3 training.

Exists solely to give SB3Agent a Gym surface during training.
Remove by: deleting this file and sb3_agent.py.
"""

import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Discrete

from aintelope.environments.gridworld import GridworldEnv


def flatten_obs(obs: dict) -> np.ndarray:
    """Combine vision + interoception into a single (C+N, H, W) float32 array."""
    vision = obs["vision"]
    interoception = obs["interoception"]
    H, W = vision.shape[1], vision.shape[2]
    intero_layers = np.broadcast_to(
        interoception[:, None, None], (len(interoception), H, W)
    ).copy()
    return np.concatenate([vision, intero_layers], axis=0)


class GridworldGymWrapper(gym.Env):
    """Single-agent Gymnasium wrapper over GridworldEnv.

    Requires env.manifesto to be populated (i.e. env.reset() called first).
    Reward is interoception[0] (food contact signal).
    """

    def __init__(self, env: GridworldEnv, agent_id: str):
        self._env = env
        self._agent_id = agent_id
        shapes = env.manifesto["observation_shapes"]
        C, H, W = shapes["vision"]
        N = shapes["interoception"][0]
        self.observation_space = Box(
            low=0.0, high=1.0, shape=(C + N, H, W), dtype=np.float32
        )
        self.action_space = Discrete(len(env.manifesto["action_space"]))
        self.render_mode = None

    def reset(self, seed=None, options=None):
        kwargs = {} if seed is None else {"seed": seed}
        obs_dict, _ = self._env.reset(**kwargs)
        return flatten_obs(obs_dict[self._agent_id]), {}

    def step(self, action):
        obs_dict, state = self._env.step_parallel({self._agent_id: {"action": action}})
        obs = obs_dict[self._agent_id]
        return (
            flatten_obs(obs),
            float(obs["interoception"][0]),
            state["dones"][self._agent_id],
            False,
            {},
        )
