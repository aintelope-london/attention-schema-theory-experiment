import numpy as np
from aintelope.environments.abstract_env import AbstractEnv

# ── Tile indices ───────────────────────────────────────────────────────────────
FLOOR, WALL, PREDATOR, FOOD, FOOD_UNRIPE, FOOD_ROTTEN, ROCK, WATER, BUSH = (
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
)
_N_BASE = 9  # agent tiles begin here

_NEXT_STAGE = {
    BUSH: FOOD_UNRIPE,
    FOOD_UNRIPE: FOOD,
    FOOD: FOOD_ROTTEN,
    FOOD_ROTTEN: BUSH,
}
_FOOD_STAGES = (BUSH, FOOD_UNRIPE, FOOD, FOOD_ROTTEN)
_PASSABLE = {FLOOR, FOOD, FOOD_UNRIPE, FOOD_ROTTEN, PREDATOR}
_FOOD_REWARD = {FOOD}

# ── Orientation ────────────────────────────────────────────────────────────────
_N, _E, _S, _W = (-1, 0), (0, 1), (1, 0), (0, -1)
_DEFAULT_FACING = _N

_ROT = {_N: 0, _W: 1, _S: 2, _E: 3}


def _cw(d):
    return (d[1], -d[0])


def _ccw(d):
    return (-d[1], d[0])


def _flip(d):
    return (-d[0], -d[1])


def _agents_from_cfg(cfg):
    return sorted(cfg.agent_params.agents.keys())


def _scramble_seed(seed: int) -> int:
    seed = seed & 0xFFFFFFFFFFFFFFFF
    seed = ((seed ^ (seed >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    seed = ((seed ^ (seed >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return (seed ^ (seed >> 31)) & 0xFFFFFFFFFFFFFFFF


def _blit_roi(viewport_mask, position, facing, board_shape):
    """Blit a viewport-space ROI mask to absolute board coordinates.

    Undoes the facing rotation before blitting — the ROI component operates
    on the rotated viewport, so we rotate back before placing on the board.
    """
    h, w = board_shape
    v = viewport_mask.shape[0]
    half = v // 2
    r, c = position
    unrotated = np.rot90(viewport_mask.astype(np.float32), k=(4 - _ROT[facing]) % 4)
    absolute = np.zeros((h, w), dtype=np.float32)
    b_r, b_c = r - half, c - half
    s_r, d_r = max(0, b_r), max(0, -b_r)
    s_c, d_c = max(0, b_c), max(0, -b_c)
    eh = min(v - d_r, h - s_r)
    ew = min(v - d_c, w - s_c)
    absolute[s_r : s_r + eh, s_c : s_c + ew] = unrotated[d_r : d_r + eh, d_c : d_c + ew]
    return absolute


class GridworldEnv(AbstractEnv):
    """Minimal randomised gridworld."""

    def __init__(self, cfg):
        self._cfg = cfg
        self.agents = _agents_from_cfg(cfg)
        self.layers = [
            "floor",
            "wall",
            "predator",
            "food",
            "food_unripe",
            "food_rotten",
            "rock",
            "water",
            "bush",
        ] + self.agents
        self._ripening_period = cfg.env_params.get("ripening", 0)
        self._ripe_randomness = cfg.env_params.get("ripe_randomness", 0)
        self._initial_food_tile = FOOD_UNRIPE if self._ripening_period > 0 else FOOD
        self._manifesto = None
        self.state = {}
        self._board = None
        self._facing = {}
        self._positions = {}
        self._predator_cells = set()
        self._food_age = {}
        self._step_rng = None

    # ── AbstractEnv contract ───────────────────────────────────────────────────

    def reset(self, **kwargs):
        seed = _scramble_seed(self._cfg.run.seed + kwargs.get("seed", 0))
        layout_rng = np.random.default_rng(seed)
        self._step_rng = np.random.default_rng(seed + 1)
        self._place_board(layout_rng)
        self._manifesto = self._build_manifesto()
        self._refresh_state(dones={aid: False for aid in self.agents}, roi_masks={})
        return self._observations(), self.state

    def step_parallel(self, actions):
        interoceptions, ate_food = {}, {}
        order = list(actions.keys())
        self._step_rng.shuffle(order)
        for aid in order:
            name = self._manifesto["action_names"][actions[aid]["action"]]
            interoceptions[aid], ate_food[aid] = getattr(self, name)(aid)
        self._ripen()
        termination = self._cfg.env_params.get("termination", None)
        dones = {aid: termination == "food" and ate_food[aid] for aid in self.agents}
        roi_masks = {
            aid: actions[aid]["roi"] for aid in self.agents if "roi" in actions[aid]
        }
        self._refresh_state(dones, roi_masks)
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

    @property
    def render_manifest(self):
        return {layer: layer for layer in self.layers}

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
        return np.zeros(2, np.float32), False

    # ── Movement ──────────────────────────────────────────────────────────────

    def _move(self, aid, direction):
        r, c = self._positions[aid]
        dr, dc = direction
        nr, nc = r + dr, c + dc
        tile = int(self._board[nr, nc])
        if tile not in _PASSABLE:
            intero = np.zeros(2, np.float32)
            intero[1] = -1.0
            return intero, False
        intero = np.zeros(2, np.float32)
        self._board[r, c] = PREDATOR if (r, c) in self._predator_cells else FLOOR
        self._positions[aid] = (nr, nc)
        if tile in _FOOD_REWARD:
            intero[0] = 1.0
        elif tile == PREDATOR:
            intero[1] = 1.0
        if tile in _PASSABLE - {FLOOR, PREDATOR}:
            self._food_age.pop((nr, nc), None)
        self._board[nr, nc] = _N_BASE + self.agents.index(aid)
        return intero, tile in _FOOD_REWARD

    # ── Ripening ──────────────────────────────────────────────────────────────

    def _next_threshold(self, current_age):
        return (
            current_age
            + self._ripening_period
            + (
                int(self._step_rng.integers(0, self._ripe_randomness + 1))
                if self._ripe_randomness
                else 0
            )
        )

    def _ripen(self):
        if not self._ripening_period:
            return
        for pos in list(self._food_age):
            self._food_age[pos][0] += 1
            age, threshold = self._food_age[pos]
            if age < threshold:
                continue
            next_stage = _NEXT_STAGE.get(int(self._board[pos]))
            if next_stage is not None:
                self._board[pos] = next_stage
                self._food_age[pos] = [age, self._next_threshold(age)]

    # ── Board initialisation ──────────────────────────────────────────────────

    def _place_board(self, rng):
        layout = self._cfg.env_params.get("map_layout", None)
        if layout:
            self._place_board_from_layout(layout)
        else:
            self._place_board_random(rng)

    def _place_board_random(self, rng):
        sz = self._cfg.env_params.map_size
        h = w = sz + 2
        self._board = np.full((h, w), FLOOR, dtype=np.int8)
        self._board[0, :] = self._board[-1, :] = WALL
        self._board[:, 0] = self._board[:, -1] = WALL
        self._predator_cells = set()
        self._food_age = {}
        self._facing = {aid: _DEFAULT_FACING for aid in self.agents}
        self._positions = {}

        cells = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
        rng.shuffle(cells)
        idx = 0

        for i, aid in enumerate(self.agents):
            pos = cells[idx]
            idx += 1
            self._positions[aid] = pos
            self._board[pos] = _N_BASE + i

        for tile_name, params in self._cfg.env_params.objects.items():
            tile_int = (
                self._initial_food_tile
                if tile_name == "food"
                else self.layers.index(tile_name)
            )
            for _ in range(params.count):
                pos = cells[idx]
                idx += 1
                self._board[pos] = tile_int
                if tile_int == PREDATOR:
                    self._predator_cells.add(pos)
                if tile_int in _FOOD_STAGES:
                    self._food_age[pos] = [0, self._next_threshold(0)]

    _LAYOUT_CHARS = {
        ".": FLOOR,
        "#": WALL,
        "P": PREDATOR,
        "F": FOOD,
        "u": FOOD_UNRIPE,
        "x": FOOD_ROTTEN,
        "r": ROCK,
        "w": WATER,
        "b": BUSH,
    }

    def _place_board_from_layout(self, layout):
        rows = [r for r in layout.strip().splitlines()]
        h, w = len(rows), max(len(r) for r in rows)
        self._board = np.full((h, w), FLOOR, dtype=np.int8)
        self._predator_cells = set()
        self._food_age = {}
        self._facing = {aid: _DEFAULT_FACING for aid in self.agents}
        self._positions = {}
        agent_iter = iter(enumerate(self.agents))

        for r, row in enumerate(rows):
            for c, ch in enumerate(row):
                if ch in self._LAYOUT_CHARS:
                    tile = self._LAYOUT_CHARS[ch]
                    self._board[r, c] = tile
                    if tile == PREDATOR:
                        self._predator_cells.add((r, c))
                    if tile in _FOOD_STAGES:
                        self._food_age[(r, c)] = [0, self._next_threshold(0)]
                elif ch == "A":
                    i, aid = next(agent_iter)
                    self._positions[aid] = (r, c)
                    self._board[r, c] = _N_BASE + i

    # ── State ─────────────────────────────────────────────────────────────────

    def _refresh_state(self, dones, roi_masks):
        cube = np.stack(
            [self._board == i for i in range(len(self.layers))], axis=0
        ).astype(np.float32)
        food_tiles = np.isin(self._board, list(_FOOD_STAGES))
        food_ys, food_xs = np.where(food_tiles)
        h, w = self._board.shape
        mask = np.zeros((len(self.agents), h, w), dtype=np.float32)
        for i, aid in enumerate(self.agents):
            if aid in roi_masks:
                mask[i] = _blit_roi(
                    roi_masks[aid], self._positions[aid], self._facing[aid], (h, w)
                )
        self.state = {
            "board": cube,
            "layers": self.layers,
            "directions": dict(self._facing),
            "dones": dones,
            "scores": {aid: {} for aid in self.agents},
            "agent_positions": dict(self._positions),
            "food_position": (int(food_ys[0]), int(food_xs[0]))
            if len(food_ys)
            else None,
            "mask": mask,
        }

    # ── Observation encoding ───────────────────────────────────────────────────

    def _observations(self, interoceptions=None):
        interoceptions = interoceptions or {}
        encode = getattr(self, f"_encode_{self._cfg.env_params.observation_format}")
        return {
            aid: {
                "vision": encode(aid),
                "interoception": interoceptions.get(aid, np.zeros(2, np.float32)),
            }
            for aid in self.agents
        }

    def _encode_boolean_cube(self, aid):
        radius = self._cfg.env_params.observation_radius
        r, c = self._positions[aid]
        h, w = self._board.shape
        v = 2 * radius + 1

        patch = np.full((v, v), WALL, dtype=np.int8)
        r0, c0 = r - radius, c - radius
        sr, sc = max(0, -r0), max(0, -c0)
        er, ec = min(v, h - r0), min(v, w - c0)
        patch[sr:er, sc:ec] = self._board[
            max(0, r0) : max(0, r0) + (er - sr),
            max(0, c0) : max(0, c0) + (ec - sc),
        ]
        patch = np.rot90(patch, k=_ROT[self._facing[aid]])
        return np.stack([patch == i for i in range(len(self.layers))], axis=0).astype(
            np.float32
        )

    # ── Manifesto ─────────────────────────────────────────────────────────────

    def _build_manifesto(self):
        radius = self._cfg.env_params.observation_radius
        v = 2 * radius + 1
        actions = list(self._cfg.env_params.actions)
        return {
            "layers": self.layers,
            "observation_shapes": {
                "vision": (len(self.layers), v, v),
                "interoception": (2,),
            },
            "action_space": actions,
            "action_names": {i: name for i, name in enumerate(actions)},
            "food_ind": self.layers.index("food"),
        }
