"""Savanna environment wrapper.

Translates savanna's interface to the AbstractEnv contract.
This is the ONLY file that imports from savanna_safetygrid.
"""

import numpy as np

from aintelope.environments.abstract_env import AbstractEnv
from aintelope.environments.savanna_safetygrid import (
    SavannaGridworldParallelEnv,
    SavannaGridworldSequentialEnv,
    INFO_AGENT_OBSERVATION_LAYERS_CUBE,
    INFO_AGENT_OBSERVATION_LAYERS_ORDER,
    INFO_AGENT_INTEROCEPTION_VECTOR,
    AGENT_CHR1,
    AGENT_CHR2,
)


from pettingzoo import ParallelEnv
from aintelope.utils.roi import compute_roi


# Savanna agent characters in order — maps agent index to layer key
_AGENT_CHRS = [AGENT_CHR1, AGENT_CHR2]

# Directions enum from gridworld engine → (row_delta, col_delta)
_DIRECTION_VECTORS = {
    0: (0, -1),  # LEFT
    1: (0, 1),  # RIGHT
    2: (-1, 0),  # UP
    3: (1, 0),  # DOWN
}

_ENV_CLASS = {
    "parallel": SavannaGridworldParallelEnv,
    "sequential": SavannaGridworldSequentialEnv,
}


class SavannaWrapper(AbstractEnv, ParallelEnv):
    def __init__(self, cfg):
        self._cfg = cfg
        self._mode = cfg.env_params.mode
        self._env = _ENV_CLASS[self._mode](cfg=cfg)
        self._manifesto = None

    def __getattr__(self, name):
        """Delegate unknown attributes to the inner env for legacy compatibility."""
        return getattr(self._env, name)

    @property
    def _scalarize_rewards(self):
        return self._env._scalarize_rewards

    @_scalarize_rewards.setter
    def _scalarize_rewards(self, value):
        self._env._scalarize_rewards = value

    # ── AbstractEnv contract ──────────────────────────────────────────

    def reset(self, **kwargs):
        raw_obs, raw_infos = self._env.reset(**kwargs)
        self._manifesto = self._build_manifesto(raw_infos)
        self._augment_infos(raw_infos)
        observations = self._apply_roi(raw_obs)
        return observations, raw_infos

    def step(self, actions):
        """PettingZoo ParallelEnv interface — used by SB3 training loop.
        HACK: includes ROI because SB3 bypasses experiments.py.
        Remove when SB3 agents are deprecated."""
        raw_obs, raw_scores, terminateds, truncateds, raw_infos = self._env.step(
            actions
        )
        self._augment_infos(raw_infos)
        observations = self._apply_roi(raw_obs)
        return observations, raw_scores, terminateds, truncateds, raw_infos

    def step_parallel(self, actions):
        raw_obs, raw_scores, terminateds, truncateds, raw_infos = self._env.step(
            actions
        )
        self._augment_infos(raw_infos)
        return raw_obs, raw_scores, terminateds, truncateds, raw_infos

    def step_sequential(self, actions):
        # TODO: sequential stepping — iterate agents internally,
        # collecting intermediate observations. For now, delegate to
        # parallel (savanna sequential env handles ordering).
        raise NotImplementedError("Sequential wrapper pending design decision")

    def board_state(self):
        board = next(iter(self._env.observe_absolute_bitmaps().values()))
        layer_order = next(iter(self._env.relative_observation_layers_order().values()))
        return board, layer_order

    @property
    def score_dimensions(self):
        from aintelope.config.config_utils import get_score_dimensions

        return get_score_dimensions(self._cfg)

    def observation_space(self, agent_id):
        orig = self._env.observation_space(agent_id)
        roi_mode = self._cfg.agent_params.roi_mode
        if roi_mode:  # any active roi_mode adds agent ROI layers
            from gymnasium.spaces import Box

            n_roi_layers = self._env.max_num_agents
            new_shape = (orig.shape[0] + n_roi_layers,) + orig.shape[1:]
            return Box(low=0, high=1, shape=new_shape, dtype=orig.dtype)
        return orig

    @property
    def observation_spaces(self):
        return {aid: self.observation_space(aid) for aid in self.possible_agents}

    def action_space(self, agent_id):
        return self._env.action_space(agent_id)

    @property
    def max_num_agents(self):
        return self._env.max_num_agents

    @property
    def agents(self):
        return self._env.agents

    @property
    def manifesto(self):
        return self._manifesto

    # ── Internal ──────────────────────────────────────────────────────

    def _build_manifesto(self, raw_infos):
        sample = next(iter(raw_infos.values()))
        layers = list(sample[INFO_AGENT_OBSERVATION_LAYERS_ORDER])
        return {"layers": layers}

    def _augment_infos(self, raw_infos):
        """Add generic keys to savanna's info dicts. Original keys preserved."""
        positions = self._read_absolute_positions()
        directions = self._read_directions()
        for agent_id, raw_info in raw_infos.items():
            raw_info["position"] = positions[agent_id]
            raw_info["direction"] = directions[agent_id]
            raw_info["raw_observation"] = (
                raw_info[INFO_AGENT_OBSERVATION_LAYERS_CUBE],
                raw_info[INFO_AGENT_INTEROCEPTION_VECTOR],
            )

    def _read_absolute_positions(self):
        """Read all agent absolute positions from the global board. One read."""
        board = next(iter(self._env.observe_absolute_bitmaps().values()))
        layer_order = list(
            next(iter(self._env.relative_observation_layers_order().values()))
        )
        positions = {}
        for i in range(self._env.max_num_agents):
            agent_id = f"agent_{i}"
            agent_chr = _AGENT_CHRS[i]
            layer_idx = layer_order.index(agent_chr)
            ys, xs = np.where(board[layer_idx])
            positions[agent_id] = (int(ys[0]), int(xs[0]))
        return positions

    def _read_directions(self):
        """Read all agent directions from the gridworld engine. SSOT."""
        sprites = self._env._env.environment_data["agent_sprite"]
        directions = {}
        for i in range(self._env.max_num_agents):
            agent_id = f"agent_{i}"
            agent_chr = self._env.agent_name_mapping[agent_id]
            direction_enum = sprites[agent_chr].observation_direction
            directions[agent_id] = _DIRECTION_VECTORS[direction_enum.value]
        return directions

    def _apply_roi(self, observations):
        """Apply ROI augmentation to observations."""
        roi_mode = self._cfg.agent_params.roi_mode
        radius = self._cfg.env_params.render_agent_radius
        positions = self._read_absolute_positions()
        directions = self._read_directions()
        return compute_roi(
            observations,
            positions,
            directions,
            roi_mode,
            radius,
        )
