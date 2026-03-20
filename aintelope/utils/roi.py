"""Attention Region of Interest computation.

compute_roi is now a passthrough -- ROI is applied inside the agent's ROI
component (agents/model/roi.py) and no longer needs to be computed externally.
Kept for SB3 compatibility in savanna_wrapper.step().

_cone_mask is retained for use in tests.
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


def compute_roi(observations, infos, cfg):
    """Passthrough -- ROI is now applied inside the agent's ROI component.

    Returns observations unchanged and an empty absolute mask array.
    Signature preserved for SB3 compatibility in savanna_wrapper.step().
    """
    board_h, board_w = next(iter(infos.values()))["board_shape"]
    return observations, np.zeros((0, board_h, board_w), dtype=bool)
