# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""ROI (Region of Interest) attention component.

Sits in the connectome DAG as a pre-filter for vision. Triggered by any
downstream component that lists this component's id in its inputs.

Mechanics:
  - Reads activations["internal_action"] written by the strategy component in the
    PREVIOUS step (persists across the update boundary by design).
  - Updates roi_state from that value.
  - Applies darkening to vision[:-1] (original channels) outside the cone.
  - Writes own boolean mask into vision[-1] (the ROI layer slot the env
    always appends as zeros) and into activations[self.component_id].

When ROI is absent from the architecture entirely (roi_mode: null), the env
wrapper still appends a zero channel as vision[-1] for consistent observation
shapes. ROI.activate() is simply never called — no vestigial branches needed.

n_internal_actions: 3 is declared on the architecture entry in config, not
as a class attribute. Model.fill_plans reads it generically via
entry.get("n_internal_actions", 0) without mentioning ROI by name. Any
future component that contributes internal action slots follows the same
pattern.

Module-level constants — promote to config when stable.
"""

import numpy as np
from aintelope.agents.model.component import Component

DARKENING_FACTOR = 0.1

_CONE_HALF_ARC = np.pi / 4  # +- 45 deg cone opening
_ROI_TURN_STEP = np.pi / 4  # 45 deg per action step

# left / stay / right -- indexed by internal_action integer
_ROI_ANGLE_DELTAS = (-_ROI_TURN_STEP, 0.0, _ROI_TURN_STEP)


def _cone_mask(h, w, center_r, center_c, angle, radius):
    """Boolean 90 deg cone mask in viewport coordinates.

    angle=0.0 -> north (up in viewport).
    Agent's own cell always included.
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

    n_internal_actions is declared on the architecture entry in config as:
        roi:
          type: ROI
          inputs: [internal_action]
          n_internal_actions: 3

    Model.fill_plans reads this generically — ROI is not mentioned by name
    anywhere in model.py. Future components follow the same pattern.

    roi_state for cone: {"angle": float} -- 0.0 = north in viewport space.
    Future shapes extend roi_state naturally (circle adds radius, offsets).
    """

    def __init__(self, context):
        self.component_id = context["component_id"]
        self.inputs = context["inputs"]
        self.radius = context["cfg"].env_params.render_agent_radius
        self.roi_state = {"angle": 0.0}

    def reset(self):
        self.roi_state = {"angle": 0.0}

    def activate(self, activations):
        # internal_action from the PREVIOUS step persists in activations.
        # Absent on step 0 and after episode reset -- angle stays at 0.0.
        internal_action = activations.get("internal_action")
        if internal_action is not None:
            self.roi_state["angle"] += _ROI_ANGLE_DELTAS[internal_action]

        vision = activations["vision"]  # (C, H, W); vision[-1] is ROI layer slot
        h, w = vision.shape[1], vision.shape[2]
        mask = _cone_mask(h, w, h // 2, w // 2, self.roi_state["angle"], self.radius)

        # Two independent operations on the same mask -- mask is never derived
        # from vision and neither operation feeds the other.
        vision[:-1] *= np.where(mask, 1.0, DARKENING_FACTOR)  # darken original channels
        vision[-1] = mask.astype(np.float32)  # write ROI layer slot
        activations[self.component_id] = mask

    def update(self, signals=None):
        return None
