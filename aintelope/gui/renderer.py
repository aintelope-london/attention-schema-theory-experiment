"""Rendering for episode playback and animation export.

Contains:
    TILE_INDEX — keyword → tile index within a tileset sprite strip.
    Tileset — reads a sprite strip BMP, tile size parsed from filename.
    StateRenderer — agnostic, composites tiles onto a PIL Image.
    SavannaInterpreter — maps savanna env layer keys to keywords.

To support a new environment, add an interpreter that maps env layer
keys to tile keywords via a MANIFEST dict, and implements interpret(state).
"""

import re
from pathlib import Path
from PIL import Image

from aintelope.environments.savanna_safetygrid import (
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
)


# =============================================================================
# Tile keywords — canonical vocabulary the renderer understands
# =============================================================================

VOID = "VOID"
WALL = "WALL"
DANGER = "DANGER"
PREDATOR = "PREDATOR"
DRINK = "DRINK"
DRINK_SMALL = "DRINK_SMALL"
FOOD = "FOOD"
FOOD_SMALL = "FOOD_SMALL"
GOLD = "GOLD"
SILVER = "SILVER"
FLOOR = "FLOOR"
AGENTS = [f"AGENT_{i}" for i in range(10)]

# Keyword → tile index within the sprite strip
TILE_INDEX = {
    VOID: 0,
    WALL: 1,
    DANGER: 2,
    PREDATOR: 3,
    DRINK: 4,
    DRINK_SMALL: 5,
    FOOD: 6,
    FOOD_SMALL: 7,
    GOLD: 8,
    SILVER: 9,
    **{agent: 10 + i for i, agent in enumerate(AGENTS)},
    FLOOR: 20,
}


# =============================================================================
# Tileset — agnostic sprite strip reader
# =============================================================================


class Tileset:
    """Reads a sprite strip. Tile size parsed from '_WxH' in filename."""

    def __init__(self, path):
        self.image = Image.open(path)
        w, h = re.search(r"(\d+)x(\d+)", Path(path).stem).groups()
        self.tile_w, self.tile_h = int(w), int(h)

    def tile(self, index):
        x = index * self.tile_w
        return self.image.crop((x, 0, x + self.tile_w, self.tile_h))


def find_tileset(directory=None):
    """Find the first BMP tileset in the given directory (default: gui/)."""
    directory = directory or Path(__file__).parent
    return str(next(Path(directory).glob("*.bmp")))


# =============================================================================
# Interpreters — one per environment
# =============================================================================


class SavannaInterpreter:
    """Maps savanna env layer keys to renderer keywords."""

    MANIFEST = {
        GAP_CHR: VOID,
        WALL_CHR: WALL,
        DANGER_TILE_CHR: DANGER,
        PREDATOR_NPC_CHR: PREDATOR,
        DRINK_CHR: DRINK,
        SMALL_DRINK_CHR: DRINK_SMALL,
        FOOD_CHR: FOOD,
        SMALL_FOOD_CHR: FOOD_SMALL,
        GOLD_CHR: GOLD,
        SILVER_CHR: SILVER,
        AGENT_CHR1: AGENTS[0],
        AGENT_CHR2: AGENTS[1],
        ALL_AGENTS_LAYER: None,
    }

    def interpret(self, state):
        """Extract renderable layers from env state.

        Args:
            state: (cube, layer_order) tuple from states.csv.

        Returns:
            (cube, layers, floor) where layers is [(index, keyword), ...].
        """
        cube, layer_order = state
        layers = [
            (idx, self.MANIFEST[key])
            for idx, key in enumerate(layer_order[: cube.shape[0]])
            if key in self.MANIFEST and self.MANIFEST[key] is not None
        ]
        return cube, layers, FLOOR


# =============================================================================
# Renderer — agnostic
# =============================================================================


class StateRenderer:
    """Composites tiles onto a PIL Image from a 3D boolean cube.

    Speaks only keywords. Unknown keywords default to FLOOR.
    """

    def __init__(self, tileset):
        self.tileset = tileset
        self._cache = {}

    def _get_tile(self, keyword):
        if keyword not in self._cache:
            idx = TILE_INDEX.get(keyword, TILE_INDEX[FLOOR])
            self._cache[keyword] = self.tileset.tile(idx)
        return self._cache[keyword]

    def render(self, cube, layers, floor):
        """Produce a PIL Image from observation data.

        Args:
            cube: 3D bool array [n_layers, height, width].
            layers: list of (layer_index, keyword), last = highest priority.
            floor: keyword for cells where no layer is active.

        Returns:
            PIL.Image.Image
        """
        tw, th = self.tileset.tile_w, self.tileset.tile_h
        height, width = cube.shape[1], cube.shape[2]
        img = Image.new("RGB", (width * tw, height * th))

        floor_tile = self._get_tile(floor)
        for y in range(height):
            for x in range(width):
                img.paste(floor_tile, (x * tw, y * th))

        for layer_idx, keyword in layers:
            tile = self._get_tile(keyword)
            for y in range(height):
                for x in range(width):
                    if cube[layer_idx, y, x]:
                        img.paste(tile, (x * tw, y * th))

        return img
