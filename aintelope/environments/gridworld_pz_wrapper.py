# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""PettingZoo ParallelEnv wrapper around GridworldEnv.

Exists solely to give SB3Agent a PZ surface it can pass to
SingleAgentZooToGymAdapter during training. Not used by any other consumer.

Remove by: deleting this file and sb3_agent.py.
"""

import numpy as np
from gymnasium.spaces import Box, Discrete
from pettingzoo import ParallelEnv

from aintelope.environments.gridworld import GridworldEnv


def flatten_obs(obs: dict) -> np.ndarray:
    """Combine vision + interoception into a single (C+N, H, W) float32 array.

    This is the single definition of the observation format shared between
    the wrapper (training path) and SB3Agent (test path).
    """
    vision = obs["vision"]
    interoception = obs["interoception"]
    H, W = vision.shape[1], vision.shape[2]
    intero_layers = np.broadcast_to(
        interoception[:, None, None], (len(interoception), H, W)
    ).copy()
    return np.concatenate([vision, intero_layers], axis=0)


class GridworldPZWrapper(ParallelEnv):
    """Thin PettingZoo ParallelEnv over GridworldEnv.

    Observations are flattened to (C+N, H, W) via flatten_obs().
    Reward is scalar: interoception[0] (food contact signal).
    Requires env.manifesto to be populated (i.e. env.reset() called first).
    """

    metadata = {}

    def __init__(self, env: GridworldEnv):
        self._env = env
        self.possible_agents = list(env.agents)
        self.agents = list(env.agents)

    @property
    def num_agents(self):
        return len(self.agents)

    def observation_space(self, agent_id):
        shapes = self._env.manifesto["observation_shapes"]
        C, H, W = shapes["vision"]
        N = shapes["interoception"][0]
        return Box(low=0.0, high=1.0, shape=(C + N, H, W), dtype=np.float32)

    @property
    def observation_spaces(self):
        return {aid: self.observation_space(aid) for aid in self.possible_agents}

    def action_space(self, agent_id):
        return Discrete(len(self._env.manifesto["action_space"]))

    @property
    def action_spaces(self):
        return {aid: self.action_space(aid) for aid in self.possible_agents}

    def reset(self, seed=None, options=None):
        kwargs = {} if seed is None else {"seed": seed}
        obs_dict, _ = self._env.reset(**kwargs)
        self.agents = list(self._env.agents)
        observations = {aid: flatten_obs(obs) for aid, obs in obs_dict.items()}
        infos = {aid: {} for aid in obs_dict}
        return observations, infos

    def step(self, actions):
        wrapped = {aid: {"action": a} for aid, a in actions.items()}
        obs_dict, state = self._env.step_parallel(wrapped)
        dones = state["dones"]
        observations = {aid: flatten_obs(obs) for aid, obs in obs_dict.items()}
        rewards = {aid: float(obs_dict[aid]["interoception"][0]) for aid in obs_dict}
        terminateds = dict(dones)
        truncateds = {aid: False for aid in obs_dict}
        infos = {aid: {} for aid in obs_dict}
        self.agents = [aid for aid, done in dones.items() if not done]
        return observations, rewards, terminateds, truncateds, infos
