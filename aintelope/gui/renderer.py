"""Rendering for episode playback and animation export.

Contains:
    StateRenderer  — agnostic, composites tiles onto a PIL Image.
    overlay        — composites colored boolean masks onto a rendered image.
    Interpreter    — maps env layer names to keywords via a render_manifest dict.

To support a new environment, implement render_manifest on the env class
returning a {layer_name: keyword} dict. No changes needed here.
To add a tile, drop a PNG into gui/tiles/ and add its name to TILES.
"""

from pathlib import Path

import numpy as np
from PIL import Image

# =============================================================================
# TILES — canonical tile registry and draw order (bottom to top).
# Each entry is the exact stem of its PNG file in gui/tiles/.
# =============================================================================

TILES = (
    "floor",
    "wall",
    "water_small",
    "water",
    "rock",
    "bush",
    "food_unripe",
    "food",
    "food_rotten",
    "danger",
    "predator",
    "agent_0",
    "agent_1",
    "agent_2",
    "agent_3",
)

FLOOR = "floor"

_TILE_DIR = Path(__file__).parent / "tiles"


# =============================================================================
# Overlay — agnostic mask compositing
# =============================================================================


def overlay(image, masks, colors, alpha):
    """Composite colored boolean masks onto a rendered image.

    Args:
        image: PIL.Image.Image (RGB).
        masks: bool ndarray [N, H, W] — grid-resolution masks.
        colors: sequence of (R, G, B) tuples, one per mask.
        alpha: int 0–255, overlay transparency.

    Returns:
        PIL.Image.Image (RGBA).
    """
    base = image.convert("RGBA")
    arr = np.zeros((base.height, base.width, 4), dtype=np.uint8)
    th = base.height // masks.shape[1]
    tw = base.width // masks.shape[2]

    for i in range(masks.shape[0]):
        pixel_mask = masks[i].repeat(th, axis=0).repeat(tw, axis=1).astype(bool)
        arr[pixel_mask] = (*colors[i], alpha)

    return Image.alpha_composite(base, Image.fromarray(arr, "RGBA"))


# =============================================================================
# Interpreter — driven by env's render_manifest
# =============================================================================


class Interpreter:
    """Maps env layer names to renderer keywords via a render_manifest dict."""

    def __init__(self, manifest):
        self._manifest = manifest
        self._priority = {kw: i for i, kw in enumerate(TILES)}

    def interpret(self, state):
        """Extract renderable layers from env state.

        Args:
            state: (board_cube, layer_names) tuple.

        Returns:
            (cube, layers, floor_keyword) where layers is [(index, keyword), ...].
        """
        cube, layer_order = state
        layers = sorted(
            [
                (idx, self._manifest[name])
                for idx, name in enumerate(layer_order[: cube.shape[0]])
                if name in self._manifest
            ],
            key=lambda pair: self._priority.get(pair[1], -1),
        )
        return cube, layers, FLOOR


# =============================================================================
# Renderer — agnostic
# =============================================================================


class StateRenderer:
    """Composites tiles onto a PIL Image from a 3D boolean cube."""

    def __init__(self):
        self._tiles = {
            p.stem: Image.open(p).convert("RGB") for p in _TILE_DIR.glob("*.png")
        }
        ref = next(iter(self._tiles.values()))
        self.tile_w, self.tile_h = ref.size

    def _tile(self, keyword):
        return self._tiles.get(keyword, self._tiles[FLOOR])

    def render(self, cube, layers, floor):
        """Produce a PIL Image from observation data.

        Args:
            cube: 3D bool array [n_layers, height, width].
            layers: list of (layer_index, keyword), last = highest priority.
            floor: keyword for cells where no layer is active.

        Returns:
            PIL.Image.Image
        """
        tw, th = self.tile_w, self.tile_h
        height, width = cube.shape[1], cube.shape[2]
        img = Image.new("RGB", (width * tw, height * th))

        floor_tile = self._tile(floor)
        for y in range(height):
            for x in range(width):
                img.paste(floor_tile, (x * tw, y * th))

        for layer_idx, keyword in layers:
            tile = self._tile(keyword)
            for y in range(height):
                for x in range(width):
                    if cube[layer_idx, y, x]:
                        img.paste(tile, (x * tw, y * th))

        return img