"""Minimal randomised MOMA gridworld.

Config keys consumed (all under env_params unless noted):
    cfg.run.seed                -- base RNG seed; per-episode seed = base + episode index
    cfg.env_params.map_size     -- interior play-area side length (walls added silently)
    cfg.env_params.objects      -- {tile_name: {count: N}}  e.g. {food: {count: 3}}
    cfg.env_params.actions      -- ordered list of action strings
    cfg.env_params.observation_radius  -- viewport half-size (viewport = 2r+1 square)
    cfg.env_params.observation_format  -- encoder key; add _encode_<key> to extend
    cfg.agent_params.agents     -- agent_* keys enumerate agents (SSOT for agent list)

Tile layer order — LAYERS is the canonical reference for all channel indexing.
Layer i in the board cube corresponds to LAYERS[i]:
    0  floor
    1  wall
    2  predator
    3  food
    4+ one layer per agent (index = 4 + agent index in sorted agent list)

Board cube shape: (len(LAYERS), map_size+2, map_size+2)  — float32 boolean per tile.

Interoception channels (reset to zero each step, set by tile collision):
    [0]  food eaten this step
    [1]  predator contact this step

Observation format "boolean_cube":
    (C, H, W) float32 boolean cube, agent-centric viewport, rotated so the
    agent's facing direction is always "up" in the frame.

Action semantics (relative orientation):
    forward   -- move one step in facing direction
    backward  -- reverse facing 180°, move one step in new direction
    left      -- rotate facing 90° CCW, move one step in new direction
    right     -- rotate facing 90° CW, move one step in new direction
    wait      -- no-op

Passability: floor, food, predator are passable.
             wall and other agents are not.
Predator: passable (triggers interoception[1]) and persists on board after contact.
Food: passable (triggers interoception[0]) and is consumed (removed from board).
"""

import numpy as np
from aintelope.environments.abstract_env import AbstractEnv

# ── Tile indices ───────────────────────────────────────────────────────────────
FLOOR, WALL, PREDATOR, FOOD = 0, 1, 2, 3
_N_BASE = 4  # number of base tile types; agent tiles begin at _N_BASE

_PASSABLE = {FLOOR, FOOD, PREDATOR}

# ── Orientation ────────────────────────────────────────────────────────────────
_N, _E, _S, _W = (-1, 0), (0, 1), (1, 0), (0, -1)
_DEFAULT_FACING = _N

# CCW 90° rotations needed to bring each facing direction "up" in the viewport
_ROT = {_N: 0, _W: 1, _S: 2, _E: 3}


def _cw(d):   return (d[1], -d[0])
def _ccw(d):  return (-d[1], d[0])
def _flip(d): return (-d[0], -d[1])


def _agents_from_cfg(cfg):
    """Sorted canonical agent ID list. SSOT: cfg.agent_params.agents keys."""
    return sorted(cfg.agent_params.agents.keys())


class GridworldEnv(AbstractEnv):
    """Minimal randomised gridworld. No PettingZoo / gym dependencies."""

    def __init__(self, cfg):
        self._cfg = cfg
        self.agents = _agents_from_cfg(cfg)
        # Layers: base tiles then one layer per agent in sorted order
        self.layers = ["floor", "wall", "predator", "food"] + self.agents
        self._manifesto = None
        self.state = {}
        # Mutable episode state
        self._board = None           # (H, W) int8
        self._facing = {}            # {agent_id: (dr, dc)}
        self._positions = {}         # {agent_id: (r, c)}
        self._predator_cells = set() # cells that always hold a predator
        self._step_rng = None

    # ── AbstractEnv contract ───────────────────────────────────────────────────

    def reset(self, **kwargs):
        """Place board, return (observations, state).

        Accepts keyword 'seed' (episode index) which offsets cfg.run.seed.
        """
        seed = self._cfg.run.seed + kwargs.get("seed", 0)
        layout_rng = np.random.default_rng(seed)
        self._step_rng = np.random.default_rng(seed + 1)
        self._place_board(layout_rng)
        self._manifesto = self._build_manifesto()
        self._refresh_state(dones={aid: False for aid in self.agents})
        return self._observations(), self.state

    def step_parallel(self, actions):
        """Execute all agent actions simultaneously (shuffled order for fairness).

        Args:
            actions: {agent_id: {"action": int, ...}}

        Returns:
            observations: {agent_id: {"vision": ndarray, "interoception": ndarray}}
            state:        canonical world snapshot dict (see module docstring)
        """
        interoceptions = {}
        order = list(actions.keys())
        self._step_rng.shuffle(order)
        for aid in order:
            name = self._manifesto["action_names"][actions[aid]["action"]]
            interoceptions[aid] = getattr(self, name)(aid)
        self._refresh_state(dones={aid: False for aid in self.agents})
        return self._observations(interoceptions), self.state

    def step_sequential(self, actions):
        raise NotImplementedError

    @property
    def manifesto(self):
        return self._manifesto

    @property
    def score_dimensions(self):
        return []

    @property
    def max_num_agents(self):
        return len(self.agents)

    # ── Action methods ─────────────────────────────────────────────────────────

    def forward(self, aid):
        return self._move(aid, self._facing[aid])

    def backward(self, aid):
        self._facing[aid] = _flip(self._facing[aid])
        return self._move(aid, self._facing[aid])

    def left(self, aid):
        self._facing[aid] = _ccw(self._facing[aid])
        return self._move(aid, self._facing[aid])

    def right(self, aid):
        self._facing[aid] = _cw(self._facing[aid])
        return self._move(aid, self._facing[aid])

    def wait(self, aid):
        return np.zeros(2, np.float32)

    # ── Movement ──────────────────────────────────────────────────────────────

    def _move(self, aid, direction):
        """Attempt move in direction. Returns interoception delta (2,)."""
        r, c = self._positions[aid]
        dr, dc = direction
        nr, nc = r + dr, c + dc
        tile = self._board[nr, nc]
        if tile not in _PASSABLE:
            return np.zeros(2, np.float32)
        intero = np.zeros(2, np.float32)
        # Restore vacated cell
        self._board[r, c] = PREDATOR if (r, c) in self._predator_cells else FLOOR
        self._positions[aid] = (nr, nc)
        if tile == FOOD:
            intero[0] = 1.0
        elif tile == PREDATOR:
            intero[1] = 1.0
        self._board[nr, nc] = _N_BASE + self.agents.index(aid)
        return intero

    # ── Board initialisation ──────────────────────────────────────────────────

    def _place_board(self, rng):
        sz = self._cfg.env_params.map_size
        h = w = sz + 2
        self._board = np.full((h, w), FLOOR, dtype=np.int8)
        self._board[0, :] = self._board[-1, :] = WALL
        self._board[:, 0] = self._board[:, -1] = WALL
        self._predator_cells = set()
        self._facing = {aid: _DEFAULT_FACING for aid in self.agents}
        self._positions = {}

        cells = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
        rng.shuffle(cells)
        idx = 0

        for i, aid in enumerate(self.agents):
            pos = cells[idx]; idx += 1
            self._positions[aid] = pos
            self._board[pos] = _N_BASE + i

        for tile_name, params in self._cfg.env_params.objects.items():
            tile_int = self.layers.index(tile_name)
            for _ in range(params.count):
                pos = cells[idx]; idx += 1
                self._board[pos] = tile_int
                if tile_int == PREDATOR:
                    self._predator_cells.add(pos)

    # ── State ─────────────────────────────────────────────────────────────────

    def _refresh_state(self, dones):
        """Rebuild self.state from current board. Called after every reset/step."""
        cube = np.stack(
            [self._board == i for i in range(len(self.layers))], axis=0
        ).astype(np.float32)
        food_ys, food_xs = np.where(self._board == FOOD)
        self.state = {
            "board":           cube,
            "layers":          self.layers,
            "directions":      dict(self._facing),
            "dones":           dones,
            "scores":          {aid: {} for aid in self.agents},
            "agent_positions": dict(self._positions),
            "food_position":   (int(food_ys[0]), int(food_xs[0])) if len(food_ys) else None,
        }

    # ── Observation encoding ───────────────────────────────────────────────────

    def _observations(self, interoceptions=None):
        interoceptions = interoceptions or {}
        encode = getattr(self, f"_encode_{self._cfg.env_params.observation_format}")
        return {
            aid: {
                "vision":        encode(aid),
                "interoception": interoceptions.get(aid, np.zeros(2, np.float32)),
            }
            for aid in self.agents
        }

    def _encode_boolean_cube(self, aid):
        """Agent-centric viewport boolean cube. Facing direction is always 'up'."""
        radius = self._cfg.env_params.observation_radius
        r, c = self._positions[aid]
        h, w = self._board.shape
        v = 2 * radius + 1

        patch = np.full((v, v), WALL, dtype=np.int8)
        r0, c0 = r - radius, c - radius
        sr, sc = max(0, -r0), max(0, -c0)
        er, ec = min(v, h - r0), min(v, w - c0)
        patch[sr:er, sc:ec] = self._board[
            max(0, r0):max(0, r0) + (er - sr),
            max(0, c0):max(0, c0) + (ec - sc),
        ]
        patch = np.rot90(patch, k=_ROT[self._facing[aid]])
        return np.stack(
            [patch == i for i in range(len(self.layers))], axis=0
        ).astype(np.float32)

    # ── Manifesto ─────────────────────────────────────────────────────────────

    def _build_manifesto(self):
        radius = self._cfg.env_params.observation_radius
        v = 2 * radius + 1
        actions = list(self._cfg.env_params.actions)
        return {
            "layers":             self.layers,
            "observation_shapes": {
                "vision":        (len(self.layers), v, v),
                "interoception": (2,),
            },
            "action_space":  actions,
            "action_names":  {i: name for i, name in enumerate(actions)},
            "food_ind":      self.layers.index("food"),
        }