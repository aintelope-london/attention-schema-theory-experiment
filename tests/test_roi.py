"""Tests for aintelope.utils.roi — cone geometry, directions, integration."""

import numpy as np
import pytest
from omegaconf import OmegaConf

from aintelope.utils.roi import _cone_mask, compute_roi


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
# 3. compute_roi integration
# ═════════════════════════════════════════════════════════════════════════


def _make_cfg(roi_mode="cone", radius=2):
    return OmegaConf.create(
        {
            "agent_params": {"roi_mode": roi_mode},
            "env_params": {"render_agent_radius": radius},
        }
    )


def _make_obs_infos(board_shape, agent_positions, agent_directions, vision_layers=3):
    """Build mock observations and infos dicts for compute_roi."""
    vh, vw = board_shape  # For simplicity: viewport = board size
    observations = {}
    infos = {}
    for aid, pos in agent_positions.items():
        observations[aid] = {
            "vision": np.zeros((vision_layers, vh, vw), dtype=np.float32),
            "interoception": np.zeros((2,), dtype=np.float32),
        }
        infos[aid] = {
            "position": pos,
            "direction": agent_directions[aid],
            "board_shape": board_shape,
        }
    return observations, infos


def test_compute_roi_output_shape():
    """compute_roi appends N_agents layers to each agent's vision."""
    cfg = _make_cfg()
    obs, infos = _make_obs_infos(
        board_shape=(5, 5),
        agent_positions={"a0": (2, 2), "a1": (0, 0)},
        agent_directions={"a0": (-1, 0), "a1": (1, 0)},
        vision_layers=3,
    )
    result, absolute = compute_roi(obs, infos, cfg)

    # Vision should have 3 original + 2 ROI layers = 5
    assert result["a0"]["vision"].shape[0] == 5
    assert result["a1"]["vision"].shape[0] == 5
    # Interoception unchanged
    assert result["a0"]["interoception"].shape == (2,)
    # Absolute masks: 2 agents × 5 × 5
    assert absolute.shape == (2, 5, 5)


def test_compute_roi_absolute_mask_matches_geometry():
    """Absolute mask should match direct _cone_mask call."""
    cfg = _make_cfg(radius=2)
    obs, infos = _make_obs_infos(
        board_shape=(5, 5),
        agent_positions={"agent_0": (2, 2)},
        agent_directions={"agent_0": (-1, 0)},
    )
    _, absolute = compute_roi(obs, infos, cfg)

    direct = _cone_mask(5, 5, 2, 2, -1, 0, 2)
    assert not np.any(
        absolute[0] ^ direct
    ), "Absolute mask should match direct geometry call"


def test_compute_roi_passthrough_none():
    """roi_mode=None → observations unchanged, empty absolute masks."""
    cfg = _make_cfg(roi_mode=None)
    obs, infos = _make_obs_infos(
        board_shape=(5, 5),
        agent_positions={"a0": (2, 2)},
        agent_directions={"a0": (-1, 0)},
        vision_layers=3,
    )
    result, absolute = compute_roi(obs, infos, cfg)

    assert result["a0"]["vision"].shape[0] == 3, "No layers added when roi_mode=None"
    assert absolute.shape == (0, 5, 5)


def test_compute_roi_viewport_crop():
    """On a larger board, viewport crop should extract the correct region."""
    board_h, board_w = 9, 9
    cfg = _make_cfg(radius=2)
    # Agent at (4, 4), viewport is 5x5 (2*radius+1), centered on agent
    # Vision shape must match viewport, not board
    vh, vw = 5, 5
    obs = {
        "agent_0": {
            "vision": np.zeros((3, vh, vw), dtype=np.float32),
            "interoception": np.zeros((2,), dtype=np.float32),
        }
    }
    infos = {
        "agent_0": {
            "position": (4, 4),
            "direction": (-1, 0),
            "board_shape": (board_h, board_w),
        }
    }

    result, absolute = compute_roi(obs, infos, cfg)

    # Absolute mask is on the 9x9 board
    assert absolute.shape == (1, 9, 9)
    # The cone on the board should be centered at (4,4) facing UP
    direct = _cone_mask(9, 9, 4, 4, -1, 0, 2)
    assert not np.any(absolute[0] ^ direct)

    # Viewport ROI layer should be the 5x5 crop around (4,4)
    roi_layer = result["agent_0"]["vision"][3].astype(bool)  # first ROI layer
    crop = absolute[0, 2:7, 2:7]  # rows 2-6, cols 2-6
    assert not np.any(
        roi_layer ^ crop
    ), f"Viewport crop mismatch:\n{roi_layer.astype(int)}\nexpected:\n{crop.astype(int)}"


def test_compute_roi_two_agents_see_each_other():
    """Each agent's observation should contain both agents' ROI masks."""
    cfg = _make_cfg(radius=2)
    obs, infos = _make_obs_infos(
        board_shape=(7, 7),
        agent_positions={"agent_0": (3, 3), "agent_1": (3, 5)},
        agent_directions={"agent_0": (-1, 0), "agent_1": (0, -1)},
        vision_layers=3,
    )
    # Override vision to be 7x7 (viewport = full board for simplicity)
    for aid in obs:
        obs[aid]["vision"] = np.zeros((3, 7, 7), dtype=np.float32)

    result, absolute = compute_roi(obs, infos, cfg)

    # Each agent gets 2 ROI layers (one per agent)
    assert result["agent_0"]["vision"].shape[0] == 5
    assert result["agent_1"]["vision"].shape[0] == 5

    # agent_0's observation layer 3 = agent_0's mask, layer 4 = agent_1's mask
    # (sorted by agent id)
    a0_sees_a0 = result["agent_0"]["vision"][3]
    a0_sees_a1 = result["agent_0"]["vision"][4]
    assert a0_sees_a0[3, 3], "agent_0 should see its own ROI at its position"
    assert a0_sees_a1[3, 5], "agent_0 should see agent_1's ROI at agent_1's position"


def test_compute_roi_edge_agent():
    """Agent at board corner — no crash, mask clips to grid."""
    cfg = _make_cfg(radius=2)
    obs = {
        "agent_0": {
            "vision": np.zeros((3, 5, 5), dtype=np.float32),
            "interoception": np.zeros((2,), dtype=np.float32),
        }
    }
    infos = {
        "agent_0": {
            "position": (0, 0),
            "direction": (-1, 0),
            "board_shape": (5, 5),
        }
    }
    result, absolute = compute_roi(obs, infos, cfg)
    assert absolute[0, 0, 0], "Agent's own cell always lit"
    assert absolute.shape == (1, 5, 5)


if __name__ == "__main__":
    pytest.main([__file__])
