"""Attention Region of Interest computation.

Appends per-agent boolean attention masks to the vision component
of each agent's observation dict. Geometry functions are stateless
spatial math.
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


def compute_masks(grid_shape, positions, directions, roi_mode, radius):
    """Compute boolean masks on a grid.

    Args:
        grid_shape: (H, W)
        positions:  {id: (row, col)}
        directions: {id: (drow, dcol)}
        roi_mode:   geometry key, e.g. "cone"
        radius:     vision radius

    Returns:
        ndarray [N, H, W] bool — one mask per id, sorted by key.
        (0, H, W) when positions is empty.
    """
    h, w = grid_shape
    shape_fn = _REGISTRY[roi_mode]
    ids = sorted(positions.keys())
    masks = np.empty((len(ids), h, w), dtype=bool)
    for i, mid in enumerate(ids):
        r, c = positions[mid]
        dr, dc = directions[mid]
        masks[i] = shape_fn(h, w, r, c, dr, dc, radius)
    return masks


def append_roi_layers(vision_obs, positions, directions, roi_mode, radius):
    """Append per-agent ROI mask layers to vision ndarrays.

    Low-level function operating on bare ndarrays. Used by environment
    wrappers that handle legacy observation formats.
    Passthrough when roi_mode is None.

    Args:
        vision_obs: {agent_id: ndarray [layers, H, W]}
        positions:  {agent_id: (row, col)}
        directions: {agent_id: (drow, dcol)}
        roi_mode:   geometry key, or None for passthrough
        radius:     vision radius

    Returns:
        {agent_id: ndarray [layers + N_masks, H, W]}
    """
    if roi_mode is None:
        return vision_obs
    agent_ids = sorted(vision_obs.keys())
    viewport_h, viewport_w = next(iter(vision_obs.values())).shape[1:]
    half_h, half_w = viewport_h // 2, viewport_w // 2

    result = {}
    for obs_id in agent_ids:
        obs = vision_obs[obs_id]
        obs_r, obs_c = positions[obs_id]

        vp_positions = {
            aid: (
                positions[aid][0] - obs_r + half_h,
                positions[aid][1] - obs_c + half_w,
            )
            for aid in agent_ids
        }

        roi_layers = compute_masks(
            (viewport_h, viewport_w), vp_positions, directions, roi_mode, radius
        )
        result[obs_id] = np.concatenate([obs, roi_layers], axis=0)

    return result


def compute_roi(observations, infos, roi_mode, radius):
    """Apply ROI to observations. Passthrough when roi_mode is None.

    Args:
        observations: {agent_id: {"vision": ndarray, "interoception": ndarray, ...}}
        infos:        {agent_id: {"position": (r,c), "direction": (dr,dc), ...}}
        roi_mode:     geometry key (e.g. "cone"), or None for passthrough
        radius:       vision radius

    Returns:
        observations with ROI layers appended to the vision component.
        Unchanged when roi_mode is None.
    """
    if roi_mode is None:
        return observations

    positions = {aid: infos[aid]["position"] for aid in infos}
    directions = {aid: infos[aid]["direction"] for aid in infos}
    vision_obs = {aid: obs["vision"] for aid, obs in observations.items()}

    roi_vision = append_roi_layers(vision_obs, positions, directions, roi_mode, radius)

    return {
        aid: {
            **observations[aid],
            "vision": roi_vision[aid],
        }
        for aid in observations
    }
