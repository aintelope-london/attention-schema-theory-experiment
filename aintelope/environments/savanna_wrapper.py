"""Savanna environment wrapper.

Translates savanna's PettingZoo interface to the AbstractEnv contract.
This is the ONLY file that imports from savanna_safetygrid.

Layout seed computation lives here — the single permitted location.
SB3 training callbacks are the second permitted location (special permission,
documented in DOCUMENTATION.md).
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
    INFO_REWARD_DICT,
    AGENT_CHR1,
    AGENT_CHR2,
    ALL_AGENTS_LAYER,
    DANGER_TILE_CHR,
    DRINK_CHR,
    FOOD_CHR,
    GAP_CHR,
    GOLD_CHR,
    PREDATOR_NPC_CHR,
    SILVER_CHR,
    SMALL_DRINK_CHR,
    SMALL_FOOD_CHR,
    WALL_CHR,
    Actions,
)

from pettingzoo import ParallelEnv
from aintelope.utils.roi import compute_roi


_AGENT_CHRS = [AGENT_CHR1, AGENT_CHR2]

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

_OWN_ATTRS = frozenset(
    {
        "_cfg",
        "_mode",
        "_env",
        "_manifesto",
        "_sb3_training",
        "_infos",
        "_dones",
        "_mask",
        "_n_agents",
        "state",
    }
)


def _blit_viewport_to_absolute(roi_mask, position, absolute_roi):
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
    Single definition — used by both experiment.py and this wrapper.
    """
    result = {}
    for aid, obs in observations.items():
        vision = obs["vision"]
        interoception = obs["interoception"]
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
        self._n_agents = sum(1 for k in cfg.agent_params.agents)
        with open_dict(cfg):
            cfg.env_params.combine_interoception_and_vision = False
        self._env = _ENV_CLASS[self._mode](cfg=cfg)
        self._manifesto = None
        self._sb3_training = False
        self._infos = {}
        self._dones = {}
        self.state = {}
        self._mask = np.zeros(
            (self._n_agents, cfg.env_params.map_height, cfg.env_params.map_width),
            dtype=np.float32,
        )

    def __getattr__(self, name):
        return getattr(self._env, name)

    def __setattr__(self, name, value):
        if name in _OWN_ATTRS or not hasattr(self, "_env"):
            super().__setattr__(name, value)
        else:
            setattr(self._env, name, value)

    # ── AbstractEnv contract ──────────────────────────────────────────

    def reset(self, **kwargs):
        """Reset. Accepts keyword 'seed' (episode index) for layout randomisation."""
        if self._pre_reset_callback2 is not None:
            # SB3 training: callback owns layout seed computation (permitted exception)
            raw_obs, raw_infos = self._env.reset(**kwargs)
        else:
            layout_seed = self._layout_seed(kwargs.get("seed", 0))
            raw_obs, raw_infos = self._env.reset(
                options={"env_layout_seed": layout_seed}
            )

        self._manifesto = self._build_manifesto(raw_obs, raw_infos)
        self._update_state(raw_infos)
        observations = self._to_dict_obs(raw_obs)
        board_h, board_w = self.state["board"].shape[1:]
        self._mask = np.zeros((self._n_agents, board_h, board_w), dtype=np.float32)
        self.state["mask"] = self._mask
        observations = self._append_roi_layer(observations)

        if self._sb3_training:
            return combine_obs_sb3(observations), raw_infos
        return observations, self.state

    def step(self, actions):
        """PettingZoo ParallelEnv interface — SB3 training only.
        Remove when SB3 agents are deprecated."""
        raw_obs, raw_scores, terminateds, truncateds, raw_infos = self._env.step(
            actions
        )
        self._update_state(raw_infos, terminateds, truncateds, raw_scores)
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
        """Canonical parallel step.

        Args:
            actions: {agent_id: {"action": int, ...}}

        Returns:
            observations, state
        """
        env_actions = {aid: a["action"] for aid, a in actions.items()}
        raw_obs, raw_scores, terminateds, truncateds, raw_infos = self._env.step(
            env_actions
        )
        self._update_state(raw_infos, terminateds, truncateds, raw_scores)
        observations = self._to_dict_obs(raw_obs)

        board_h, board_w = self.state["board"].shape[1:]
        self._mask = np.zeros((self._n_agents, board_h, board_w), dtype=bool)
        for i, aid in enumerate(sorted(actions.keys())):
            mask = actions[aid].get("roi")
            if mask is not None:
                _blit_viewport_to_absolute(
                    mask, self._infos[aid]["position"], self._mask[i]
                )
        self.state["mask"] = self._mask
        observations = self._append_roi_layer(observations)
        return observations, self.state

    def step_sequential(self, actions):
        raise NotImplementedError

    @property
    def manifesto(self):
        return self._manifesto

    @property
    def score_dimensions(self):
        from aintelope.config.config_utils import get_score_dimensions

        return get_score_dimensions(self._cfg)

    @property
    def render_manifest(self):
        return {
            GAP_CHR: "VOID",
            WALL_CHR: "WALL",
            DANGER_TILE_CHR: "DANGER",
            PREDATOR_NPC_CHR: "PREDATOR",
            DRINK_CHR: "DRINK",
            SMALL_DRINK_CHR: "DRINK_SMALL",
            FOOD_CHR: "FOOD",
            SMALL_FOOD_CHR: "FOOD_SMALL",
            GOLD_CHR: "GOLD",
            SILVER_CHR: "SILVER",
            **{f"agent_{i}": f"AGENT_{i}" for i in range(self._n_agents)},
        }

    def observation_space(self, agent_id):
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

    # ── Layout seed (single permitted location for canonical path) ────

    def _layout_seed(self, seed):
        """Compute layout seed from episode index. Encapsulates all seed logic."""
        repeat_len = self._cfg.env_params.env_layout_seed_repeat_sequence_length
        modulo = self._cfg.env_params.env_layout_seed_modulo
        layout_seed = int(seed / repeat_len) if repeat_len > 0 else seed
        return layout_seed % modulo if modulo > 0 else layout_seed

    # ── State ─────────────────────────────────────────────────────────

    def _update_state(
        self, raw_infos, terminateds=None, truncateds=None, raw_scores=None
    ):
        """Update self._infos, self._dones, and rebuild self.state."""
        terminateds = terminateds or {}
        truncateds = truncateds or {}
        raw_scores = raw_scores or {}

        board = next(iter(self._env.observe_absolute_bitmaps().values())).copy()
        layer_order = list(
            next(iter(self._env.relative_observation_layers_order().values()))
        )
        board_shape = board.shape[1:]
        directions = self._read_directions()

        # Extract positions using raw savanna chars
        positions = {}
        for i in range(self._env.max_num_agents):
            agent_id = f"agent_{i}"
            layer_idx = layer_order.index(_AGENT_CHRS[i])
            ys, xs = np.where(board[layer_idx])
            positions[agent_id] = (int(ys[0]), int(xs[0]))

        food_pos = None
        if FOOD_CHR in layer_order:
            ys, xs = np.where(board[layer_order.index(FOOD_CHR)])
            food_pos = (int(ys[0]), int(xs[0])) if len(ys) > 0 else None

        # Translate savanna chars to canonical agent ids before storing
        for i, chr in enumerate(_AGENT_CHRS):
            if chr in layer_order:
                layer_order[layer_order.index(chr)] = f"agent_{i}"

        self._dones = {
            aid: (terminateds.get(aid, False) or truncateds.get(aid, False))
            for aid in raw_infos
        }

        for agent_id, info in raw_infos.items():
            info["position"] = positions[agent_id]
            info["direction"] = directions[agent_id]
            info["board_shape"] = board_shape
            info["food_position"] = food_pos
            info["raw_observation"] = (
                info[INFO_AGENT_OBSERVATION_LAYERS_CUBE],
                info[INFO_AGENT_INTEROCEPTION_VECTOR],
            )

        self._infos = raw_infos

        self.state = {
            "board": board,
            "layers": layer_order,
            "directions": directions,
            "dones": dict(self._dones),
            "scores": {
                aid: info.get(INFO_REWARD_DICT, {}) for aid, info in raw_infos.items()
            },
            "agent_positions": positions,
            "food_position": food_pos,
        }

    # ── Observation format ────────────────────────────────────────────

    def _to_dict_obs(self, raw_obs):
        return {
            agent_id: {"vision": obs[0], "interoception": obs[1]}
            for agent_id, obs in raw_obs.items()
        }

    def _append_roi_layer(self, observations):
        for aid, obs in observations.items():
            pos = self._infos[aid]["position"]
            layers = [
                _crop_absolute_to_viewport(
                    self._mask[i],
                    pos,
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
            "agent_layers": {
                f"agent_{i}": _AGENT_CHRS[i] for i in range(self._n_agents)
            },
        }

    # ── Internal ──────────────────────────────────────────────────────

    def _read_directions(self):
        sprites = self._env._env.environment_data["agent_sprite"]
        directions = {}
        for i in range(self._env.max_num_agents):
            agent_id = f"agent_{i}"
            agent_chr = self._env.agent_name_mapping[agent_id]
            direction_enum = sprites[agent_chr].observation_direction
            directions[agent_id] = _DIRECTION_VECTORS[direction_enum.value]
        return directions
