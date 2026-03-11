# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""ROI (Region of Interest) attention component.

Sits in the connectome DAG as a pre-filter for vision. Triggered by any
downstream component that lists this component's id in its inputs.

Mechanics:
  - Reads activations["extra_action"] written by the strategy component in the
    PREVIOUS step (persists across the update boundary by design — only
    next_* fields and "done" are cleared between steps).
  - Updates roi_state from that value.
  - Applies darkening to vision[:-1] (original channels) outside the cone.
  - Writes own boolean mask into vision[-1] (the ROI layer slot the env
    always appends as zeros) and into activations[self.component_id].

When roi_mode is null in cfg, activate() is a NOOP: vision[-1] is left as
zeros and activations[self.component_id] receives a zero mask. The component
is vestigial — present in the architecture, occupying its channel slot,
but producing no signal. This keeps observation shapes consistent across
roi_mode=cone and roi_mode=null runs so the same architecture config works
for both without modification.

Module-level constants — promote to config when stable:
"""

import numpy as np
from aintelope.agents.model.component import Component

# ── Hardcoded modes (promote to config when stable) ───────────────────
extra_action_MODE = "cone"  # future: "circle"
DARKENING_MODE = "noise"  # future: "shadow"
DARKENING_FACTOR = 0.1

_CONE_HALF_ARC = np.pi / 4  # +-45deg cone opening
_ROI_TURN_STEP = np.pi / 4  # 45deg per action step

# left / stay / right -- indexed by extra_action integer
_ROI_ANGLE_DELTAS = (-_ROI_TURN_STEP, 0.0, _ROI_TURN_STEP)


def _cone_mask(h, w, center_r, center_c, angle, radius):
    """Boolean 90deg cone mask in viewport coordinates.

    angle=0.0 -> north (up in viewport, matching agent's relative orientation).
    Increases clockwise. Agent's own cell always included.
    """
    dr_dir = -np.cos(angle)
    dc_dir = np.sin(angle)

    rows, cols = np.mgrid[0:h, 0:w]
    dr = rows - center_r
    dc = cols - center_c
    dist_sq = dr * dr + dc * dc
    dist = np.sqrt(dist_sq)

    cos_angle = (dr * dr_dir + dc * dc_dir) / np.maximum(dist, 1e-10)

    return (dist_sq == 0) | (
        (dist_sq <= radius * radius) & (cos_angle >= np.cos(_CONE_HALF_ARC) - 1e-10)
    )


class ROI(Component):
    """Attention ROI component.

    n_extra_actions is a class-level constant read by Model during init to
    extend the q_net output vector. No network plans config is needed --
    radius is read from cfg directly.

    roi_state for cone: {"angle": float} -- 0.0 = north in viewport space.
    Future shapes extend roi_state naturally (circle adds radius, offsets).
    """

    n_extra_actions = len(_ROI_ANGLE_DELTAS)  # 3: left / stay / right

    def __init__(self, context):
        self.component_id = context["component_id"]
        self.inputs = context["inputs"]
        self.cfg = context["cfg"]
        self.radius = context["cfg"].env_params.render_agent_radius
        self.roi_state = {"angle": 0.0}

    def reset(self):
        self.roi_state = {"angle": 0.0}

    def activate(self, activations):
        vision = activations["vision"]  # (C, H, W); vision[-1] is ROI layer slot
        h, w = vision.shape[1], vision.shape[2]

        if self.cfg.agent_params.roi_mode is None:
            # Vestigial: leave vision[-1] as zeros, write zero mask.
            activations[self.component_id] = np.zeros((h, w), dtype=bool)
            return

        # extra_action from the PREVIOUS step persists in activations.
        # Absent on step 0 and after episode reset -- angle stays at 0.0.
        extra_action = activations.get("extra_action")
        if extra_action is not None:
            self.roi_state["angle"] += _ROI_ANGLE_DELTAS[extra_action]

        mask = _cone_mask(h, w, h // 2, w // 2, self.roi_state["angle"], self.radius)

        # Two independent operations on the same mask -- mask is never derived
        # from vision and neither operation feeds the other.
        vision[:-1] *= np.where(mask, 1.0, DARKENING_FACTOR)  # darken original channels
        vision[-1] = mask.astype(np.float32)  # write ROI layer slot
        activations[self.component_id] = mask

    def update(self, signals=None):
        return None
