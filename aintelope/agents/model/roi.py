# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""ROI (Region of Interest) attention component.

Sits in the connectome DAG as a pre-filter for vision. Triggered by any
downstream component that lists this component's id in its inputs.

Mechanics:
  - Reads activations["internal_action"] written by the strategy component in the
    PREVIOUS step (persists across the update boundary by design).
  - Updates roi_state["angle"] from that value.
  - Applies darkening to vision[:-1] (original channels) outside the mask.
  - Writes own boolean mask into vision[-1] (the ROI layer slot the env
    always appends as zeros) and into activations[self.component_id].

When ROI is absent from the architecture entirely, the env wrapper still
appends a zero channel as vision[-1] for consistent observation shapes.
ROI.activate() is simply never called -- no vestigial branches needed.

All parameters (roi_mode, darkening_factor, turn_step, and shape-specific
fields) live in the architecture entry in models_library.yaml.  ROI reads
them from context["plans"], which model.py populates with the entry itself
for components that have no library card in cfg.models.  This is the same
mechanism any future self-parameterised component should use.

n_internal_actions is declared on the architecture entry and read generically
by model.py -- ROI does not expose it as a class attribute.

Dispatch tables _MASK_FNS / _MASK_PARAMS / _INITIAL_STATES route all
shape-specific behaviour by roi_mode without any branching in the class.

roi_state keys per mode:
  cone:   {"angle"}
  circle: {"angle", "distance", "radius"}
"""

import numpy as np
from aintelope.agents.model.component import Component


def _cone_mask(h, w, center_r, center_c, angle, radius, half_arc, **_):
    """Boolean cone mask in viewport coordinates.

    angle=0.0 -> north (up in viewport). Increases clockwise.
    half_arc: half-opening in radians (plans.half_arc).
    radius:   cone reach in cells (plans.cone_radius).
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
        (dist_sq <= radius * radius) & (cos_angle >= np.cos(half_arc) - 1e-10)
    )


def _circle_mask(h, w, center_r, center_c, angle, distance, radius, **_):
    """Circle spotlight mask in viewport coordinates.

    The spotlight center orbits the agent at `distance` cells in direction
    `angle` (0=north, clockwise).  All cells within `radius` of that center
    are illuminated.  The agent's own cell is not guaranteed to be included.

    angle:    direction to spotlight center (roi_state, controlled by actions).
    distance: orbit radius -- distance from agent to spotlight center
              (roi_state, initialised from plans.circle_distance).
    radius:   spotlight coverage radius
              (roi_state, initialised from plans.circle_radius).
    """
    circle_r = center_r - distance * np.cos(angle)
    circle_c = center_c + distance * np.sin(angle)

    rows, cols = np.mgrid[0:h, 0:w]
    dr = rows - circle_r
    dc = cols - circle_c

    return (dr * dr + dc * dc) <= radius * radius


# ── Dispatch tables ────────────────────────────────────────────────────────

# Mask function for each roi_mode.
_MASK_FNS = {
    "cone": _cone_mask,
    "circle": _circle_mask,
}

# Fixed per-mode params extracted from plans at init; passed to mask fn each step.
_MASK_PARAMS = {
    "cone": lambda plans: {
        "radius": plans.cone_radius,
        "half_arc": plans.half_arc,
    },
    "circle": lambda plans: {},  # all dynamic params live in roi_state
}

# Initial roi_state for each mode.  circle seeds distance/radius from plans
# so they are agent-controllable in future (roi_state is mutable per step).
_INITIAL_STATES = {
    "cone": lambda _plans: {"angle": 0.0},
    "circle": lambda plans: {
        "angle": 0.0,
        "distance": float(plans.circle_distance),
        "radius": float(plans.circle_radius),
    },
}


class ROI(Component):
    """Attention ROI component.

    All parameters are declared on the architecture entry in models_library.yaml:

        roi:
          type: ROI
          inputs: [internal_action]
          outputs: [roi]
          n_internal_actions: 3
          roi_mode: cone
          darkening_factor: 0.1
          half_arc: 0.7854   # cone only -- half-opening in radians (pi/4 = 45 deg)
          turn_step: 0.7854  # radians per action step (pi/4 = 45 deg)
          cone_radius: 2     # cone only -- reach in cells
          # circle_distance: 2.0  # circle only -- orbit radius
          # circle_radius: 2.0    # circle only -- spotlight coverage radius

    model.py passes the architecture entry as context["plans"] for components
    without a cfg.models library card -- ROI reads everything from there.
    """

    def __init__(self, context):
        plans = context["plans"]
        self.component_id = context["component_id"]
        self.roi_mode = plans.roi_mode
        self._darkening_factor = float(plans.darkening_factor)
        turn_step = float(plans.turn_step)
        self._angle_deltas = (-turn_step, 0.0, turn_step)
        self._mask_params = _MASK_PARAMS[self.roi_mode](plans)
        self._initial_roi_state = _INITIAL_STATES[self.roi_mode](plans)
        self.roi_state = dict(self._initial_roi_state)

    def reset(self):
        self.roi_state = dict(self._initial_roi_state)

    def activate(self, activations):
        # internal_action from the PREVIOUS step persists in activations.
        # Defaults to 1 (stay) on step 0 / after episode reset so the angle
        # accumulation is always a single unconditional statement.
        self.roi_state["angle"] += self._angle_deltas[
            activations.get("internal_action", 1)
        ]

        vision = activations["vision"]  # (C, H, W); vision[-1] is ROI layer slot
        h, w = vision.shape[1], vision.shape[2]
        mask = _MASK_FNS[self.roi_mode](
            h,
            w,
            h // 2,
            w // 2,
            **self.roi_state,
            **self._mask_params,
        )

        # Two independent operations on the same mask -- mask is never derived
        # from vision and neither operation feeds the other.
        vision[:-1] *= np.where(mask, 1.0, self._darkening_factor)  # darken outside ROI
        vision[-1] = mask.astype(np.float32)  # write ROI layer slot
        activations[self.component_id] = mask
