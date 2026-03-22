# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Savanna environment wrapper.

Translates savanna's interface to the AbstractEnv contract.
This is the ONLY file that imports from savanna_safetygrid.
"""

import numpy as np
from omegaconf import open_dict

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
    Actions,
)

from pettingzoo import ParallelEnv
from aintelope.utils.roi import compute_roi


# Savanna agent characters in order -- maps agent index to layer key
_AGENT_CHRS = [AGENT_CHR1, AGENT_CHR2]

# Directions enum from gridworld engine -> (row_delta, col_delta)
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
_OWN_ATTRS = frozenset(
    {
        "_cfg",
        "_mode",
        "_env",
        "_manifesto",
        "_sb3_training",
        "_last_infos",
        "_last_aux_mask",
        "_n_agents",
    }
)


def _blit_viewport_to_absolute(roi_mask, position, absolute_roi):
    """Blit a viewport ROI mask into absolute board coordinates in-place."""
    obs_r, obs_c = position
    vh, vw = roi_mask.shape
    half_h, half_w = vh // 2, vw // 2
    board_h, board_w = absolute_roi.shape

    b_r, b_c = obs_r - half_h, obs_c - half_w
    s_r, d_r = max(0, b_r), max(0, -b_r)
    s_c, d_c = max(0, b_c), max(0, -b_c)
    h = min(vh - d_r, board_h - s_r)
    w = min(vw - d_c, board_w - s_c)
    absolute_roi[s_r : s_r + h, s_c : s_c + w] = roi_mask[d_r : d_r + h, d_c : d_c + w]


def _crop_absolute_to_viewport(absolute_roi, position, viewport_shape):
    """Crop the absolute ROI layer into an agent's viewport."""
    obs_r, obs_c = position
    vh, vw = viewport_shape
    half_h, half_w = vh // 2, vw // 2
    board_h, board_w = absolute_roi.shape

    viewport = np.zeros((vh, vw), dtype=bool)
    b_r, b_c = obs_r - half_h, obs_c - half_w
    s_r, d_r = max(0, b_r), max(0, -b_r)
    s_c, d_c = max(0, b_c), max(0, -b_c)
    h = min(vh - d_r, board_h - s_r)
    w = min(vw - d_c, board_w - s_c)
    viewport[d_r : d_r + h, d_c : d_c + w] = absolute_roi[s_r : s_r + h, s_c : s_c + w]
    return viewport


def combine_obs_sb3(observations):
    """Convert canonical dict observations to SB3 (C+N, H, W) cubes.

    Each interoception value becomes a constant-filled spatial layer.
    Single definition -- used by both experiment.py and this wrapper."""
    result = {}
    for aid, obs in observations.items():
        vision = obs["vision"]  # (C, H, W)
        interoception = obs["interoception"]  # (N,)
        H, W = vision.shape[1], vision.shape[2]
        intero_layers = np.broadcast_to(
            interoception[:, None, None], (len(interoception), H, W)
        ).copy()
        result[aid] = np.concatenate([vision, intero_layers], axis=0)
    return result


class SavannaWrapper(AbstractEnv, ParallelEnv):
    def __init__(self, cfg):
        self._cfg = cfg
        self._mode = cfg.env_params.mode
        self._n_agents = sum(1 for k in cfg.agent_params if k.startswith("agent_"))
        # Inner env always produces separate modalities -- wrapper owns the format
        with open_dict(cfg):
            cfg.env_params.combine_interoception_and_vision = False
        self._env = _ENV_CLASS[self._mode](cfg=cfg)
        self._manifesto = None
        self._sb3_training = False
        self._last_infos = None
        self._last_aux_mask = np.zeros(
            (self._n_agents, cfg.env_params.map_height, cfg.env_params.map_width),
            dtype=np.float32,
        )

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
        observations = self._to_dict_obs(raw_obs)

        board_h, board_w = next(iter(raw_infos.values()))["board_shape"]
        self._last_aux_mask = np.zeros(
            (self._n_agents, board_h, board_w), dtype=np.float32
        )
        self._last_infos = raw_infos

        observations = self._append_roi_layer(observations, raw_infos)

        if self._sb3_training:
            return combine_obs_sb3(observations), raw_infos
        return observations, raw_infos

    def step(self, actions):
        """PettingZoo ParallelEnv interface -- used by SB3 training only.
        Adds ROI and combines to flat ndarray for SB3 compatibility.
        Remove when SB3 agents are deprecated."""
        raw_obs, raw_scores, terminateds, truncateds, raw_infos = self._env.step(
            actions
        )
        self._augment_infos(raw_infos)
        observations = self._to_dict_obs(raw_obs)
        observations, _ = compute_roi(observations, raw_infos, self._cfg)
        return (
            combine_obs_sb3(observations),
            raw_scores,
            terminateds,
            truncateds,
            raw_infos,
        )

    def step_parallel(self, actions):
        """actions: {agent_id: {"action": int, ...}}

        Each agent's "roi" viewport mask is blitted into its own absolute slot
        in _last_aux_mask, keyed by sorted agent index. All N slots are then
        cropped into each observer's viewport for the next observation.
        _last_aux_mask stays zero for agents without an active ROI component,
        giving consistent observation shape across all architectures.
        """
        env_actions = {aid: a["action"] for aid, a in actions.items()}
        raw_obs, raw_scores, terminateds, truncateds, raw_infos = self._env.step(
            env_actions
        )
        self._augment_infos(raw_infos)
        observations = self._to_dict_obs(raw_obs)

        board_h, board_w = next(iter(raw_infos.values()))["board_shape"]
        self._last_infos = raw_infos
        self._last_aux_mask = np.zeros((self._n_agents, board_h, board_w), dtype=bool)
        for i, aid in enumerate(sorted(actions.keys())):
            mask = actions[aid].get("roi")
            if mask is not None:
                _blit_viewport_to_absolute(
                    mask, raw_infos[aid]["position"], self._last_aux_mask[i]
                )
        observations = self._append_roi_layer(observations, raw_infos)

        return observations, raw_scores, terminateds, truncateds, raw_infos

    def step_sequential(self, actions):
        # TODO: sequential stepping -- iterate agents internally,
        # collecting intermediate observations. For now, delegate to
        # parallel (savanna sequential env handles ordering).
        raise NotImplementedError("Sequential wrapper pending design decision")

    def board_state(self):
        board = next(iter(self._env.observe_absolute_bitmaps().values()))
        layer_order = next(iter(self._env.relative_observation_layers_order().values()))
        return board, layer_order, self._last_aux_mask

    @property
    def score_dimensions(self):
        from aintelope.config.config_utils import get_score_dimensions

        return get_score_dimensions(self._cfg)

    def observation_space(self, agent_id):
        """Gymnasium observation space for SB3 compatibility.
        Matches the combined (C+N, H, W) cube from combine_obs_sb3.
        Vision always includes n_agents ROI slots regardless of roi_mode."""
        from gymnasium.spaces import Box

        sample_obs = self._manifesto["observation_shapes"]
        vision_shape = sample_obs["vision"]
        interoception_shape = sample_obs["interoception"]

        total_channels = vision_shape[0] + interoception_shape[0]
        H, W = vision_shape[1], vision_shape[2]
        return Box(
            low=-float("inf"),
            high=float("inf"),
            shape=(total_channels, H, W),
            dtype=float,
        )

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
        """Convert inner env's tuple observations to canonical dict format."""
        return {
            agent_id: {"vision": obs[0], "interoception": obs[1]}
            for agent_id, obs in raw_obs.items()
        }

    def _append_roi_layer(self, observations, infos):
        """Append N per-agent absolute ROI slots (cropped to viewport) to each observer.

        _last_aux_mask is zero-initialised on every reset and step, so agents
        without an active ROI component produce zero slots automatically --
        no separate zero-layer path needed.
        """
        for aid, obs in observations.items():
            layers = [
                _crop_absolute_to_viewport(
                    self._last_aux_mask[i],
                    infos[aid]["position"],
                    obs["vision"].shape[1:],
                ).astype(np.float32)[np.newaxis]
                for i in range(self._n_agents)
            ]
            obs["vision"] = np.concatenate([obs["vision"]] + layers, axis=0)
        return observations

    # ── Manifesto ─────────────────────────────────────────────────────

    def _build_manifesto(self, raw_obs, raw_infos):
        sample_info = next(iter(raw_infos.values()))
        sample_agent = next(iter(raw_infos.keys()))
        sample_obs = raw_obs[sample_agent]

        layers = list(sample_info[INFO_AGENT_OBSERVATION_LAYERS_ORDER])

        vision = sample_obs[0]
        interoception = sample_obs[1]

        # Vision always gains one ROI slot per agent -- zero when no ROI
        # component is active, mask-filled when one is. Single source of truth
        # for the agent's vision channel count.
        observation_shapes = {
            "vision": (vision.shape[0] + self._n_agents, *vision.shape[1:]),
            "interoception": interoception.shape,
        }

        action_space = list(range(self._env.action_space(sample_agent).n))
        food_ind = layers.index(FOOD_CHR) if FOOD_CHR in layers else None

        action_names = {a.value: a.name for a in Actions if a.value in action_space}

        return {
            "layers": layers,
            "observation_shapes": observation_shapes,
            "action_space": action_space,
            "action_names": action_names,
            "food_ind": food_ind,
        }

    # ── Internal ──────────────────────────────────────────────────────

    def _augment_infos(self, raw_infos):
        """Add generic keys to savanna's info dicts. Original keys preserved."""
        board = next(iter(self._env.observe_absolute_bitmaps().values()))
        layer_order = list(
            next(iter(self._env.relative_observation_layers_order().values()))
        )
        board_shape = board.shape[1:]

        positions = {}
        for i in range(self._env.max_num_agents):
            agent_id = f"agent_{i}"
            layer_idx = layer_order.index(_AGENT_CHRS[i])
            ys, xs = np.where(board[layer_idx])
            positions[agent_id] = (int(ys[0]), int(xs[0]))

        directions = self._read_directions()

        food_pos = None
        if FOOD_CHR in layer_order:
            ys, xs = np.where(board[layer_order.index(FOOD_CHR)])
            food_pos = (int(ys[0]), int(xs[0])) if len(ys) > 0 else None

        for agent_id, raw_info in raw_infos.items():
            raw_info["position"] = positions[agent_id]
            raw_info["direction"] = directions[agent_id]
            raw_info["board_shape"] = board_shape
            raw_info["food_position"] = food_pos
            raw_info["raw_observation"] = (
                raw_info[INFO_AGENT_OBSERVATION_LAYERS_CUBE],
                raw_info[INFO_AGENT_INTEROCEPTION_VECTOR],
            )

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
