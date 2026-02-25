"""Attention Region of Interest computation.

Computes per-agent boolean attention masks in absolute board coordinates,
then crops and rotates them into each observer's viewport for the NN.
"""

import numpy as np

_HALF_ARC_RAD = np.pi / 4  # 45° — half of the 90° cone


def _cone_mask(h, w, center_r, center_c, dir_r, dir_c, radius):
    """Boolean 90° cone mask on an (h, w) grid.

    Cone opens ±45° from the facing direction.
    Agent's own cell is always included.
    """
    rows, cols = np.mgrid[0:h, 0:w]
    dr = rows - center_r
    dc = cols - center_c
    dist_sq = dr * dr + dc * dc

    dir_len = np.hypot(dir_r, dir_c)
    # Zero direction (NOOP) → only agent's own cell
    nr = dir_r / max(dir_len, 1e-10)
    nc = dir_c / max(dir_len, 1e-10)

    dist = np.sqrt(dist_sq)
    cos_angle = (dr * nr + dc * nc) / np.maximum(dist, 1e-10)

    return (dist_sq <= radius * radius) & (
        (dist_sq == 0) | (cos_angle >= np.cos(_HALF_ARC_RAD) - 1e-10)
    )


_REGISTRY = {
    "cone": _cone_mask,
}


def compute_roi(observations, infos, cfg):
    """Compute ROI masks and append to observations.

    Args:
        observations: {agent_id: {"vision": ndarray, "interoception": ndarray}}
        infos:        {agent_id: {"position": (r,c), "direction": (dr,dc),
                                   "board_shape": (H,W), ...}}
        cfg:          experiment config

    Returns:
        (augmented_observations, absolute_masks)
        absolute_masks: ndarray [N, board_H, board_W] bool — for visualization.
        Empty [0, H, W] when roi_mode is None.
    """
    roi_mode = cfg.agent_params.roi_mode
    agent_ids = sorted(observations.keys())
    board_h, board_w = next(iter(infos.values()))["board_shape"]

    # Null case — identity element
    if roi_mode is None:
        return observations, np.zeros((0, board_h, board_w), dtype=bool)

    shape_fn = _REGISTRY[roi_mode]
    radius = cfg.env_params.render_agent_radius

    positions = {aid: infos[aid]["position"] for aid in agent_ids}
    directions = {aid: infos[aid]["direction"] for aid in agent_ids}

    # 1. Absolute masks on board grid
    absolute = np.empty((len(agent_ids), board_h, board_w), dtype=bool)
    for i, aid in enumerate(agent_ids):
        r, c = positions[aid]
        dr, dc = directions[aid]
        absolute[i] = shape_fn(board_h, board_w, r, c, dr, dc, radius)

    # 2. Crop into each observer's viewport
    result = {}
    for obs_id in agent_ids:
        vision = observations[obs_id]["vision"]
        vh, vw = vision.shape[1:]
        half_h, half_w = vh // 2, vw // 2
        obs_r, obs_c = positions[obs_id]

        vp = np.zeros((len(agent_ids), vh, vw), dtype=bool)
        b_r, b_c = obs_r - half_h, obs_c - half_w
        # Clipped copy from absolute board → viewport
        s_r, d_r = max(0, b_r), max(0, -b_r)
        s_c, d_c = max(0, b_c), max(0, -b_c)
        h = min(vh - d_r, board_h - s_r)
        w = min(vw - d_c, board_w - s_c)
        vp[:, d_r : d_r + h, d_c : d_c + w] = absolute[:, s_r : s_r + h, s_c : s_c + w]

        result[obs_id] = {
            **observations[obs_id],
            "vision": np.concatenate([vision, vp], axis=0),
        }

    return result, absolute
