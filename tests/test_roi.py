"""Tests for aintelope.utils.roi — cone geometry, directions, integration."""

import numpy as np
import pytest
from omegaconf import OmegaConf

from aintelope.utils.roi import _cone_mask
from aintelope.agents.model.roi import _circle_mask, ROI


# ═════════════════════════════════════════════════════════════════════════
# 1. Cone geometry — one test per cardinal direction
#    5x5 grid, center (2,2), radius 2, 90° arc (±45°)
# ═════════════════════════════════════════════════════════════════════════


def test_cone_up():
    """UP = (-1, 0): cone opens upward."""
    result = _cone_mask(5, 5, 2, 2, -1, 0, 2)
    expected = np.array(
        [
            [0, 0, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=bool,
    )
    assert not np.any(
        result ^ expected
    ), f"UP cone mismatch:\n{result.astype(int)}\nexpected:\n{expected.astype(int)}"


def test_cone_right():
    """RIGHT = (0, 1): cone opens rightward."""
    result = _cone_mask(5, 5, 2, 2, 0, 1, 2)
    expected = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 1, 1, 1],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=bool,
    )
    assert not np.any(
        result ^ expected
    ), f"RIGHT cone mismatch:\n{result.astype(int)}\nexpected:\n{expected.astype(int)}"


def test_cone_down():
    """DOWN = (1, 0): cone opens downward."""
    result = _cone_mask(5, 5, 2, 2, 1, 0, 2)
    expected = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 0, 0],
        ],
        dtype=bool,
    )
    assert not np.any(
        result ^ expected
    ), f"DOWN cone mismatch:\n{result.astype(int)}\nexpected:\n{expected.astype(int)}"


def test_cone_left():
    """LEFT = (0, -1): cone opens leftward."""
    result = _cone_mask(5, 5, 2, 2, 0, -1, 2)
    expected = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [1, 1, 1, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=bool,
    )
    assert not np.any(
        result ^ expected
    ), f"LEFT cone mismatch:\n{result.astype(int)}\nexpected:\n{expected.astype(int)}"


def test_directions_are_rotations():
    """All four cardinal cones should be 90° rotations of each other."""
    up = _cone_mask(5, 5, 2, 2, -1, 0, 2)
    right = _cone_mask(5, 5, 2, 2, 0, 1, 2)
    down = _cone_mask(5, 5, 2, 2, 1, 0, 2)
    left = _cone_mask(5, 5, 2, 2, 0, -1, 2)

    assert not np.any(np.rot90(up, k=1) ^ left), "UP rot90 should equal LEFT"
    assert not np.any(np.rot90(up, k=2) ^ down), "UP rot180 should equal DOWN"
    assert not np.any(np.rot90(up, k=3) ^ right), "UP rot270 should equal RIGHT"


# ═════════════════════════════════════════════════════════════════════════
# 2. Special cases
# ═════════════════════════════════════════════════════════════════════════


def test_cone_noop_direction():
    """Zero direction vector (0, 0) → only the agent's own cell."""
    result = _cone_mask(5, 5, 2, 2, 0, 0, 2)
    expected = np.zeros((5, 5), dtype=bool)
    expected[2, 2] = True
    assert not np.any(
        result ^ expected
    ), f"NOOP should be single cell:\n{result.astype(int)}"


def test_cone_edge_top_left():
    """Agent at (0, 0) facing UP — cone clips to grid, no crash."""
    result = _cone_mask(5, 5, 0, 0, -1, 0, 2)
    # Only the agent's own cell and maybe one cell — nothing above/left exists
    assert result[0, 0], "Agent's own cell must always be lit"
    assert result.shape == (5, 5)
    # Nothing below row 0 should be in an upward cone
    assert not np.any(result[2:, :]), "Upward cone from row 0 shouldn't reach row 2+"


def test_cone_edge_bottom_right():
    """Agent at (4, 4) facing DOWN — cone clips to grid."""
    result = _cone_mask(5, 5, 4, 4, 1, 0, 2)
    assert result[4, 4], "Agent's own cell must always be lit"
    assert not np.any(result[:3, :]), "Downward cone from row 4 shouldn't reach row 2-"


def test_cone_radius_1():
    """Radius 1: only immediate neighbors in the cone arc."""
    result = _cone_mask(5, 5, 2, 2, -1, 0, 1)
    expected = np.zeros((5, 5), dtype=bool)
    expected[2, 2] = True  # center
    expected[1, 2] = True  # one step up
    # Diagonals at dist sqrt(2) > 1, so NOT included
    assert not np.any(
        result ^ expected
    ), f"Radius 1 UP:\n{result.astype(int)}\nexpected:\n{expected.astype(int)}"


def test_cone_symmetry():
    """Cone should be symmetric about the facing direction axis."""
    up = _cone_mask(9, 9, 4, 4, -1, 0, 4)
    # Mirror left-right should be identical for UP direction
    assert not np.any(up ^ np.fliplr(up)), "UP cone should be left-right symmetric"

    right = _cone_mask(9, 9, 4, 4, 0, 1, 4)
    # Mirror top-bottom should be identical for RIGHT direction
    assert not np.any(
        right ^ np.flipud(right)
    ), "RIGHT cone should be top-bottom symmetric"


# ═════════════════════════════════════════════════════════════════════════
# 4. _circle_mask geometry
# ═════════════════════════════════════════════════════════════════════════


def test_circle_mask_center_disc():
    """distance=0 -> disc centred exactly on the agent."""
    result = _circle_mask(7, 7, 3, 3, angle=0.0, distance=0.0, radius=2.0)
    rows, cols = np.mgrid[0:7, 0:7]
    expected = (rows - 3) ** 2 + (cols - 3) ** 2 <= 4.0
    assert np.array_equal(result, expected)


def test_circle_mask_offset_north():
    """angle=0 (north) -> spotlight center at (center_r - distance, center_c)."""
    result = _circle_mask(9, 9, 4, 4, angle=0.0, distance=2.0, radius=1.0)
    rows, cols = np.mgrid[0:9, 0:9]
    expected = (rows - 2.0) ** 2 + (cols - 4.0) ** 2 <= 1.0
    assert np.array_equal(result, expected)


def test_circle_mask_offset_east():
    """angle=pi/2 (east) -> spotlight center at (center_r, center_c + distance)."""
    result = _circle_mask(9, 9, 4, 4, angle=np.pi / 2, distance=3.0, radius=1.0)
    rows, cols = np.mgrid[0:9, 0:9]
    # circle_r = 4 - 3*cos(pi/2) ~= 4, circle_c = 4 + 3*sin(pi/2) = 7
    expected = (rows - 4.0) ** 2 + (cols - 7.0) ** 2 <= 1.0
    assert np.array_equal(result, expected)


def test_circle_mask_agent_not_always_included():
    """Circle spotlight can move fully away from the agent cell."""
    # distance=3, radius=1 pointing north: spotlight center at (1, 4).
    # Agent cell (4, 4) is 3 cells away -- outside radius=1.
    result = _circle_mask(9, 9, 4, 4, angle=0.0, distance=3.0, radius=1.0)
    assert not result[
        4, 4
    ], "Agent's own cell should NOT be forced into the circle mask"


def test_circle_mask_orbit_centroid():
    """Spotlight center tracks the expected orbit position at each angle.

    Cell counts are not invariant on a discrete grid (non-integer centers shift
    which boundary cells fall inside the radius), but the centroid of lit cells
    should be within half a cell of the expected continuous center.
    """
    for angle in (0.0, np.pi / 4, np.pi / 2, np.pi, 3 * np.pi / 2):
        mask = _circle_mask(11, 11, 5, 5, angle=angle, distance=2.0, radius=1.5)
        rows, cols = np.mgrid[0:11, 0:11]
        centroid_r = rows[mask].mean()
        centroid_c = cols[mask].mean()
        expected_r = 5 - 2.0 * np.cos(angle)
        expected_c = 5 + 2.0 * np.sin(angle)
        assert (
            abs(centroid_r - expected_r) < 0.5
        ), f"angle={angle:.3f}: centroid_r={centroid_r:.2f} expected={expected_r:.2f}"
        assert (
            abs(centroid_c - expected_c) < 0.5
        ), f"angle={angle:.3f}: centroid_c={centroid_c:.2f} expected={expected_c:.2f}"


# ═════════════════════════════════════════════════════════════════════════
# 5. ROI component: reading params from plans (architecture entry)
# ═════════════════════════════════════════════════════════════════════════


def _cone_plans(**overrides):
    defaults = dict(
        roi_mode="cone",
        darkening_factor=0.1,
        half_arc=np.pi / 4,
        turn_step=np.pi / 4,
        cone_radius=2,
    )
    defaults.update(overrides)
    return OmegaConf.create(defaults)


def _circle_plans(**overrides):
    defaults = dict(
        roi_mode="circle",
        darkening_factor=0.1,
        turn_step=np.pi / 4,
        circle_distance=2.0,
        circle_radius=2.0,
    )
    defaults.update(overrides)
    return OmegaConf.create(defaults)


def _roi(plans):
    return ROI(
        {
            "plans": plans,
            "component_id": "roi",
            "inputs": ["internal_action"],
            "components": {},
            "activations": {},
            "cfg": OmegaConf.create({}),
        }
    )


def test_roi_cone_reads_plans_params():
    roi = _roi(_cone_plans(darkening_factor=0.05, half_arc=0.3, turn_step=0.1))
    assert roi._darkening_factor == pytest.approx(0.05)
    assert roi._mask_params["half_arc"] == pytest.approx(0.3)
    assert roi._angle_deltas == pytest.approx((-0.1, 0.0, 0.1))
    assert roi._mask_params["radius"] == 2


def test_roi_circle_reads_plans_params():
    roi = _roi(_circle_plans(circle_distance=1.5, circle_radius=3.0))
    assert roi.roi_state["distance"] == pytest.approx(1.5)
    assert roi.roi_state["radius"] == pytest.approx(3.0)
    assert roi.roi_state["angle"] == pytest.approx(0.0)


def test_roi_cone_reset():
    roi = _roi(_cone_plans(turn_step=np.pi / 4))
    roi.roi_state["angle"] = 2.5
    roi.reset()
    assert roi.roi_state["angle"] == pytest.approx(0.0)


def test_roi_circle_reset_restores_full_state():
    roi = _roi(_circle_plans(circle_distance=1.5, circle_radius=3.0))
    roi.roi_state["angle"] = 1.0
    roi.roi_state["distance"] = 99.0
    roi.reset()
    assert roi.roi_state["angle"] == pytest.approx(0.0)
    assert roi.roi_state["distance"] == pytest.approx(1.5)
    assert roi.roi_state["radius"] == pytest.approx(3.0)


def test_roi_activate_cone_writes_mask():
    roi = _roi(_cone_plans())
    vision = np.ones((4, 9, 9), dtype=np.float32)
    activations = {"vision": vision}
    roi.activate(activations)
    assert vision[-1].dtype == np.float32
    assert vision[-1].max() == pytest.approx(1.0)
    assert "roi" in activations


def test_roi_activate_circle_writes_mask():
    roi = _roi(_circle_plans(circle_distance=1.0, circle_radius=1.5))
    vision = np.ones((4, 9, 9), dtype=np.float32)
    activations = {"vision": vision}
    roi.activate(activations)
    assert vision[-1].max() == pytest.approx(1.0)
    assert vision[-1].sum() > 1


def test_roi_activate_darkening_zeroes_outside():
    """darkening_factor=0 -> outside-ROI cells are exactly zero."""
    roi = _roi(_cone_plans(darkening_factor=0.0))
    vision = np.ones((4, 9, 9), dtype=np.float32)
    activations = {"vision": vision}
    roi.activate(activations)
    mask = vision[-1].astype(bool)
    assert np.all(vision[0][~mask] == pytest.approx(0.0))
    assert np.all(vision[0][mask] == pytest.approx(1.0))


def test_roi_default_internal_action_is_stay():
    """Missing internal_action (step 0) defaults to stay -- angle unchanged."""
    roi = _roi(_cone_plans(turn_step=np.pi / 4))
    vision = np.ones((4, 9, 9), dtype=np.float32)
    roi.activate({"vision": vision})  # no internal_action key
    assert roi.roi_state["angle"] == pytest.approx(0.0)


if __name__ == "__main__":
    pytest.main([__file__])
