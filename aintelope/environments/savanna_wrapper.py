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
    FOOD_CHR,
)


from pettingzoo import ParallelEnv
from aintelope.utils.roi import append_roi_layers


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

# Attributes that belong to the wrapper itself; everything else proxies to _env.
_OWN_ATTRS = frozenset({"_cfg", "_mode", "_env", "_manifesto"})


class SavannaWrapper(AbstractEnv, ParallelEnv):
    def __init__(self, cfg):
        self._cfg = cfg
        self._mode = cfg.env_params.mode
        self._env = _ENV_CLASS[self._mode](cfg=cfg)
        self._manifesto = None

    def __getattr__(self, name):
        """Delegate unknown attributes to the inner env for legacy compatibility."""
        return getattr(self._env, name)

    def __setattr__(self, name, value):
        """Transparent proxy: own state stays on wrapper, everything else forwards."""
        if name in _OWN_ATTRS or not hasattr(self, "_env"):
            super().__setattr__(name, value)
        else:
            setattr(self._env, name, value)

    # ── AbstractEnv contract ──────────────────────────────────────────

    def reset(self, **kwargs):
        raw_obs, raw_infos = self._env.reset(**kwargs)
        self._manifesto = self._build_manifesto(raw_obs, raw_infos)
        self._augment_infos(raw_infos)
        observations = self._apply_roi(raw_obs)
        observations = self._to_dict_obs(observations)
        return observations, raw_infos

    def step(self, actions):
        """PettingZoo ParallelEnv interface — used by SB3 training loop.
        HACK: includes ROI because SB3 bypasses experiments.py.
        Returns legacy observation format for SB3 compatibility.
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
        observations = self._to_dict_obs(raw_obs)
        return observations, raw_scores, terminateds, truncateds, raw_infos

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

    # ── Observation format ────────────────────────────────────────────

    def _to_dict_obs(self, raw_obs):
        """Convert inner env's tuple observations to canonical dict format.
        Inner env must use combine_interoception_and_vision=False."""
        return {
            agent_id: {
                "vision": obs[0],
                "interoception": obs[1],
            }
            for agent_id, obs in raw_obs.items()
        }

    # ── Manifesto ─────────────────────────────────────────────────────

    def _build_manifesto(self, raw_obs, raw_infos):
        sample_info = next(iter(raw_infos.values()))
        sample_agent = next(iter(raw_infos.keys()))
        sample_obs = raw_obs[sample_agent]

        layers = list(sample_info[INFO_AGENT_OBSERVATION_LAYERS_ORDER])

        vision = sample_obs[0]
        interoception = sample_obs[1]

        observation_shapes = {
            "vision": vision.shape,
            "interoception": interoception.shape,
        }

        action_space = list(range(self._env.action_space(sample_agent).n))

        food_ind = layers.index(FOOD_CHR) if FOOD_CHR in layers else None

        return {
            "layers": layers,
            "observation_shapes": observation_shapes,
            "action_space": action_space,
            "food_ind": food_ind,
        }

    # ── Internal ──────────────────────────────────────────────────────

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
        """Apply ROI augmentation to observations (legacy ndarray format)."""
        roi_mode = self._cfg.agent_params.roi_mode
        radius = self._cfg.env_params.render_agent_radius
        positions = self._read_absolute_positions()
        directions = self._read_directions()
        return append_roi_layers(
            observations,
            positions,
            directions,
            roi_mode,
            radius,
        )
