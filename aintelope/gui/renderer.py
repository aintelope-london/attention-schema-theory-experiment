"""Rendering for episode playback and future animation export.

Contains:
    StateRenderer — agnostic, renders a 3D bool cube to a 2D ASCII grid.
    SavannaInterpreter — translates savanna observations into renderer input.

To support a new environment, add an interpreter class that provides
layer_chars and floor_char, and implements interpret(state).
"""

from aintelope.environments.savanna_safetygrid import (
    AGENT_CHR1,
    AGENT_CHR2,
    ALL_AGENTS_LAYER,
    DANGER_TILE_CHR,
    DRINK_CHR,
    FOOD_CHR,
    GAP_CHR,
    GOLD_CHR,
    LAYER_ORDER,
    PREDATOR_NPC_CHR,
    SILVER_CHR,
    SMALL_DRINK_CHR,
    SMALL_FOOD_CHR,
    WALL_CHR,
)


# =============================================================================
# Interpreters — one per environment
# =============================================================================


class SavannaInterpreter:
    """Translates savanna gridworld observations into renderer-ready data.

    Maps each layer key to a display character using the canonical LAYER_ORDER.
    """

    # Layer key → display character. None = skip during rendering.
    DISPLAY = {
        GAP_CHR: None,
        WALL_CHR: "#",
        DANGER_TILE_CHR: "W",
        PREDATOR_NPC_CHR: "P",
        DRINK_CHR: "D",
        SMALL_DRINK_CHR: "d",
        FOOD_CHR: "F",
        SMALL_FOOD_CHR: "f",
        GOLD_CHR: "G",
        SILVER_CHR: "S",
        AGENT_CHR1: "0",
        AGENT_CHR2: "1",
        ALL_AGENTS_LAYER: None,
    }

    FLOOR = "."

    def interpret(self, state):
        """Extract renderable layers from a savanna observation.

        Args:
            state: Deserialized (observation_cube, interoception_vector) tuple.

        Returns:
            Tuple of (cube, layers, floor) where:
                cube: 3D numpy bool array [n_layers, height, width]
                layers: list of (layer_index, display_char) in render order
                floor: character for empty cells
        """
        cube = state[0]
        layers = [
            (idx, self.DISPLAY[key])
            for idx, key in enumerate(LAYER_ORDER[: cube.shape[0]])
            if key in self.DISPLAY and self.DISPLAY[key] is not None
        ]
        return cube, layers, self.FLOOR


# =============================================================================
# Renderer — agnostic
# =============================================================================


class StateRenderer:
    """Renders a 3D boolean observation cube as a 2D ASCII grid.

    Knows nothing about environments. Receives interpreted layer data
    and stamps characters onto a grid by render priority (last wins).
    """

    def render(self, cube, layers, floor):
        """Produce ASCII grid from observation data.

        Args:
            cube: 3D bool array [n_layers, height, width].
            layers: list of (layer_index, display_char), last = highest priority.
            floor: character for cells where no layer is active.

        Returns:
            List of strings, one per grid row.
        """
        height, width = cube.shape[1], cube.shape[2]
        grid = [[floor] * width for _ in range(height)]
        for layer_idx, char in layers:
            for y in range(height):
                for x in range(width):
                    if cube[layer_idx, y, x]:
                        grid[y][x] = char
        return ["".join(row) for row in grid]
