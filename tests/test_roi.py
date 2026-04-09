"""Tests for aintelope.utils.roi — cone geometry, directions, integration."""
import numpy as np
import pytest
from omegaconf import OmegaConf
from aintelope.utils.roi import _cone_mask
from aintelope.agents.model.roi import _circle_mask, ROI


# ═════════════════════════════════════════════════════════════════════════
# 1. Cone geometry — one test per cardinal direction
# ═════════════════════════════════════════════════════════════════════════


def test_cone_up():
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
    assert not np.any(result ^ expected), f"UP cone mismatch:\n{result.astype(int)}"


def test_cone_right():
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
    assert not np.any(result ^ expected), f"RIGHT cone mismatch:\n{result.astype(int)}"


def test_cone_down():
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
    assert not np.any(result ^ expected), f"DOWN cone mismatch:\n{result.astype(int)}"


def test_cone_left():
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
    assert not np.any(result ^ expected), f"LEFT cone mismatch:\n{result.astype(int)}"


def test_directions_are_rotations():
    up = _cone_mask(5, 5, 2, 2, -1, 0, 2)
    right = _cone_mask(5, 5, 2, 2, 0, 1, 2)
    down = _cone_mask(5, 5, 2, 2, 1, 0, 2)
    left = _cone_mask(5, 5, 2, 2, 0, -1, 2)
    assert not np.any(np.rot90(up, k=1) ^ left)
    assert not np.any(np.rot90(up, k=2) ^ down)
    assert not np.any(np.rot90(up, k=3) ^ right)


# ═════════════════════════════════════════════════════════════════════════
# 2. Special cases
# ═════════════════════════════════════════════════════════════════════════


def test_cone_noop_direction():
    result = _cone_mask(5, 5, 2, 2, 0, 0, 2)
    expected = np.zeros((5, 5), dtype=bool)
    expected[2, 2] = True
    assert not np.any(result ^ expected)


def test_cone_edge_top_left():
    result = _cone_mask(5, 5, 0, 0, -1, 0, 2)
    assert result[0, 0]
    assert not np.any(result[2:, :])


def test_cone_edge_bottom_right():
    result = _cone_mask(5, 5, 4, 4, 1, 0, 2)
    assert result[4, 4]
    assert not np.any(result[:3, :])


def test_cone_radius_1():
    result = _cone_mask(5, 5, 2, 2, -1, 0, 1)
    expected = np.zeros((5, 5), dtype=bool)
    expected[2, 2] = True
    expected[1, 2] = True
    assert not np.any(result ^ expected)


def test_cone_symmetry():
    up = _cone_mask(9, 9, 4, 4, -1, 0, 4)
    assert not np.any(up ^ np.fliplr(up))
    right = _cone_mask(9, 9, 4, 4, 0, 1, 4)
    assert not np.any(right ^ np.flipud(right))


# ═════════════════════════════════════════════════════════════════════════
# 3. Circle mask geometry
# ═════════════════════════════════════════════════════════════════════════


def test_circle_mask_center_disc():
    result = _circle_mask(7, 7, 3, 3, angle=0.0, distance=0.0, radius=2.0)
    rows, cols = np.mgrid[0:7, 0:7]
    expected = (rows - 3) ** 2 + (cols - 3) ** 2 <= 4.0
    assert np.array_equal(result, expected)


def test_circle_mask_offset_north():
    result = _circle_mask(9, 9, 4, 4, angle=0.0, distance=2.0, radius=1.0)
    rows, cols = np.mgrid[0:9, 0:9]
    expected = (rows - 2.0) ** 2 + (cols - 4.0) ** 2 <= 1.0
    assert np.array_equal(result, expected)


def test_circle_mask_offset_east():
    result = _circle_mask(9, 9, 4, 4, angle=np.pi / 2, distance=3.0, radius=1.0)
    rows, cols = np.mgrid[0:9, 0:9]
    expected = (rows - 4.0) ** 2 + (cols - 7.0) ** 2 <= 1.0
    assert np.array_equal(result, expected)


def test_circle_mask_agent_not_always_included():
    result = _circle_mask(9, 9, 4, 4, angle=0.0, distance=3.0, radius=1.0)
    assert not result[4, 4]


def test_circle_mask_orbit_centroid():
    for angle in (0.0, np.pi / 4, np.pi / 2, np.pi, 3 * np.pi / 2):
        mask = _circle_mask(11, 11, 5, 5, angle=angle, distance=2.0, radius=1.5)
        rows, cols = np.mgrid[0:11, 0:11]
        centroid_r = rows[mask].mean()
        centroid_c = cols[mask].mean()
        assert abs(centroid_r - (5 - 2.0 * np.cos(angle))) < 0.5
        assert abs(centroid_c - (5 + 2.0 * np.sin(angle))) < 0.5


# ═════════════════════════════════════════════════════════════════════════
# 4. ROI component
# ═════════════════════════════════════════════════════════════════════════


def _cone_plans(**overrides):
    defaults = dict(
        roi_mode="cone",
        roi_features=["angle"],
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
        roi_features=["angle", "radius"],
        darkening_factor=0.1,
        turn_step=np.pi / 4,
        radius_step=0.5,
        circle_distance=2.0,
        circle_radius=2.0,
        circle_radius_min=0.5,
        circle_radius_max=4.5,
    )
    defaults.update(overrides)
    return OmegaConf.create(defaults)


def _radius_plans(**overrides):
    defaults = dict(
        roi_mode="circle",
        roi_features=["radius"],
        darkening_factor=0.1,
        radius_step=0.5,
        circle_distance=0.0,
        circle_radius=4.5,
        circle_radius_min=0.5,
        circle_radius_max=4.5,
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
    roi = _roi(_cone_plans(darkening_factor=0.05, turn_step=0.1))
    assert roi._darkening_factor == pytest.approx(0.05)
    assert roi._feature_deltas["angle"] == pytest.approx((-0.1, 0.0, 0.1))
    assert roi._mask_params["radius"] == 2


def test_roi_circle_reads_plans_params():
    roi = _roi(_circle_plans(circle_distance=1.5, circle_radius=3.0))
    assert roi.roi_state["distance"] == pytest.approx(1.5)
    assert roi.roi_state["radius"] == pytest.approx(3.0)
    assert roi.roi_state["angle"] == pytest.approx(0.0)


def test_roi_radius_only_reads_plans_params():
    roi = _roi(_radius_plans(circle_radius=4.0, radius_step=1.0))
    assert roi.roi_state["radius"] == pytest.approx(4.0)
    assert roi._feature_deltas["radius"] == pytest.approx((-1.0, 0.0, 1.0))
    assert roi.roi_state["angle"] == pytest.approx(0.0)  # fixed, not a feature
    assert "angle" not in roi._feature_deltas  # angle is not agent-controllable


def test_roi_cone_reset():
    roi = _roi(_cone_plans(turn_step=np.pi / 4))
    roi.roi_state["angle"] = 2.5
    roi.reset()
    assert roi.roi_state["angle"] == pytest.approx(0.0)


def test_roi_circle_reset_restores_full_state():
    roi = _roi(_circle_plans(circle_distance=1.5, circle_radius=3.0))
    roi.roi_state["angle"] = 1.0
    roi.roi_state["distance"] = 99.0
    roi.roi_state["radius"] = 0.1
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
    roi.activate({"vision": vision})
    assert vision[-1].max() == pytest.approx(1.0)
    assert vision[-1].sum() > 1


def test_roi_activate_darkening_zeroes_outside():
    roi = _roi(_cone_plans(darkening_factor=0.0))
    vision = np.ones((4, 9, 9), dtype=np.float32)
    activations = {"vision": vision}
    roi.activate(activations)
    mask = vision[-1].astype(bool)
    assert np.all(vision[0][~mask] == pytest.approx(0.0))
    assert np.all(vision[0][mask] == pytest.approx(1.0))


def test_roi_default_internal_action_is_stay():
    roi = _roi(_cone_plans(turn_step=np.pi / 4))
    vision = np.ones((4, 9, 9), dtype=np.float32)
    roi.activate({"vision": vision})
    assert roi.roi_state["angle"] == pytest.approx(0.0)


def test_roi_radius_clamped():
    roi = _roi(
        _radius_plans(
            circle_radius=4.5,
            radius_step=0.5,
            circle_radius_min=0.5,
            circle_radius_max=4.5,
        )
    )
    vision = np.ones((4, 9, 9), dtype=np.float32)
    # action 0 = shrink, repeat until clamp kicks in
    for _ in range(20):
        roi.activate({"vision": vision, "internal_action": 0})
    assert roi.roi_state["radius"] == pytest.approx(0.5)


def test_roi_angle_and_radius_independent():
    """Circle with both features: angle action doesn't move radius and vice versa."""
    roi = _roi(_circle_plans(circle_radius=2.0, turn_step=np.pi / 4, radius_step=0.5))
    vision = np.ones((4, 9, 9), dtype=np.float32)
    roi.activate({"vision": vision, "internal_action": 0})  # angle left
    assert roi.roi_state["radius"] == pytest.approx(2.0)
    roi.activate({"vision": vision, "internal_action": 3})  # radius shrink
    assert roi.roi_state["angle"] == pytest.approx(-np.pi / 4)


if __name__ == "__main__":
    pytest.main([__file__])
