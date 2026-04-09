# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np
from aintelope.agents.model.component import Component


def _cone_mask(h, w, center_r, center_c, angle, radius, half_arc, **_):
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
    circle_r = center_r - distance * np.cos(angle)
    circle_c = center_c + distance * np.sin(angle)
    rows, cols = np.mgrid[0:h, 0:w]
    dr = rows - circle_r
    dc = cols - circle_c
    return (dr * dr + dc * dc) <= radius * radius


# ── Dispatch tables ────────────────────────────────────────────────────────

_MASK_FNS = {
    "cone": _cone_mask,
    "circle": _circle_mask,
}

# Fixed params that are not agent-controllable features.
_MASK_PARAMS = {
    "cone": lambda plans: {"radius": plans.cone_radius, "half_arc": plans.half_arc},
    "circle": lambda plans: {},
}

# Non-feature roi_state entries — fixed, seeded from plans, not action-driven.
_FIXED_STATE = {
    "cone": lambda _: {},
    "circle": lambda plans: {
        "distance": float(plans.circle_distance),
        "angle": float(plans.get("circle_angle", 0.0)),
    },
}

# Per-feature: initial value and step size, both read from plans.
_FEATURE_INIT = {
    "angle": lambda _: 0.0,
    "radius": lambda plans: float(plans.circle_radius),
}
_FEATURE_STEP = {
    "angle": lambda plans: float(plans.turn_step),
    "radius": lambda plans: float(plans.radius_step),
}


class ROI(Component):
    """Attention ROI component.

    Declared on the architecture entry in models_library.yaml.
    roi_features controls which state dimensions are agent-controllable;
    each feature contributes 3 internal actions (neg, stay, pos).
    n_internal_actions is computed dynamically via ROI.n_internal_actions(plans).

    Example entry:
        roi:
          type: ROI
          inputs: [internal_action]
          outputs: [roi]
          roi_mode: circle
          roi_features: [angle, radius]
          darkening_factor: 0.1
          turn_step: 0.7854
          radius_step: 0.5
          circle_radius: 4.5
          circle_radius_min: 0.5
          circle_radius_max: 4.5
          circle_distance: 0.0
    """

    @staticmethod
    def n_internal_actions(plans):
        return len(plans.roi_features) * 3

    def __init__(self, context):
        plans = context["plans"]
        self.component_id = context["component_id"]
        self.roi_mode = plans.roi_mode
        self._darkening_factor = float(plans.darkening_factor)

        self._feature_names = list(plans.roi_features)
        self._feature_deltas = {
            f: (-_FEATURE_STEP[f](plans), 0.0, _FEATURE_STEP[f](plans))
            for f in self._feature_names
        }
        self._radius_min = float(plans.get("circle_radius_min", 0.5))
        self._radius_max = float(plans.get("circle_radius_max", float("inf")))

        self._mask_params = _MASK_PARAMS[self.roi_mode](plans)
        self._initial_roi_state = {
            **_FIXED_STATE[self.roi_mode](plans),
            **{f: _FEATURE_INIT[f](plans) for f in self._feature_names},
        }
        self.roi_state = dict(self._initial_roi_state)

    def reset(self):
        self.roi_state = dict(self._initial_roi_state)

    def activate(self, activations):
        # internal_action from the PREVIOUS step persists in activations.
        # Default 1 maps to feature_0 stay — a no-op on step 0.
        internal_action = activations.get("internal_action", 1)
        feat = self._feature_names[internal_action // 3]
        self.roi_state[feat] += self._feature_deltas[feat][internal_action % 3]
        if feat == "radius":
            self.roi_state["radius"] = np.clip(
                self.roi_state["radius"], self._radius_min, self._radius_max
            )

        vision = activations["vision"]
        h, w = vision.shape[1], vision.shape[2]
        mask = _MASK_FNS[self.roi_mode](
            h, w, h // 2, w // 2, **self.roi_state, **self._mask_params
        )
        vision[:-1] *= np.where(mask, 1.0, self._darkening_factor)
        vision[-1] = mask.astype(np.float32)
        activations[self.component_id] = mask

    def update(self, signals=None):
        return None
