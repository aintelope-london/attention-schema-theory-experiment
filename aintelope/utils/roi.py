"""Attention Region of Interest computation.

Stateless spatial geometry. No project imports.
Each shape function computes a boolean mask on a grid given
a center position, facing direction, and radius.
"""

import numpy as np


_HALF_ARC_RAD = np.pi / 4  # 45° — half of the 90° cone


def _cone_mask(h, w, center_r, center_c, dir_r, dir_c, radius):
    """Boolean cone mask on an (h, w) grid.

    Cone opens 90° (±45°) from the facing direction.
    Agent's own cell is always included.
    """
    rows, cols = np.mgrid[0:h, 0:w]
    dr = rows - center_r
    dc = cols - center_c
    dist_sq = dr * dr + dc * dc

    dir_len = np.hypot(dir_r, dir_c)
    nr, nc = dir_r / dir_len, dir_c / dir_len

    dist = np.sqrt(dist_sq)
    cos_angle = (dr * nr + dc * nc) / np.maximum(dist, 1e-10)

    return (dist_sq <= radius * radius) & (
        (dist_sq == 0) | (cos_angle >= np.cos(_HALF_ARC_RAD))
    )


_REGISTRY = {
    "cone": _cone_mask,
}


def compute_roi(observations, positions, directions, roi_mode, radius):
    """Append per-agent ROI layers to each agent's observation.

    Args:
        observations: {agent_id: ndarray [layers, H, W]}
        positions:    {agent_id: (row, col)} — absolute grid coords
        directions:   {agent_id: (drow, dcol)} — facing direction vector
        roi_mode:     geometry key, e.g. "cone"
        radius:       vision radius

    Returns:
        {agent_id: ndarray [layers + N_agents, H, W]}
        Layer order of appended ROI layers matches sorted agent_ids.
    """
    shape_fn = _REGISTRY[roi_mode]
    agent_ids = sorted(observations.keys())
    viewport_h, viewport_w = next(iter(observations.values())).shape[1:]
    half_h, half_w = viewport_h // 2, viewport_w // 2

    result = {}
    for obs_id in agent_ids:
        obs = observations[obs_id]
        obs_r, obs_c = positions[obs_id]

        roi_layers = np.empty((len(agent_ids), viewport_h, viewport_w), dtype=obs.dtype)
        for i, agent_id in enumerate(agent_ids):
            ag_r, ag_c = positions[agent_id]
            ag_dir_r, ag_dir_c = directions[agent_id]
            # Transform to observer's viewport coordinates
            rel_r = ag_r - obs_r + half_h
            rel_c = ag_c - obs_c + half_w
            roi_layers[i] = shape_fn(
                viewport_h, viewport_w, rel_r, rel_c, ag_dir_r, ag_dir_c, radius
            )

        result[obs_id] = np.concatenate([obs, roi_layers], axis=0)

    return result
