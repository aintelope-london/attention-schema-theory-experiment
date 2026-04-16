# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Abstract environment contract.

Minimal multi-agent MDP interface. The orchestration layer
(experiment.py) depends only on this contract, never on concrete
environment internals.

Canonical state schema
----------------------
Both reset() and step_parallel() return a state dict with at minimum:
    {
        "board":           ndarray (C, H, W) float32  — full layer cube,
                           one boolean channel per layer; complete world snapshot.
        "layers":          list[str]  — channel names; index matches board axis 0.
        "directions":      {agent_id: (row_delta, col_delta)}  — facing vectors.
        "dones":           {agent_id: bool}  — whether each agent is done.
        "scores":          {agent_id: dict[str, float]}  — env-reported scores per
                           dimension. Empty dict for environments where reward
                           inference is handled agent-side.
        "agent_positions": {agent_id: (row, col)}  — convenience; derivable from board
                           but provided directly to avoid layer-naming resolution
                           in consumers.
        "food_position":   (row, col) | None  — first food tile, or None.
        "mask":            ndarray (N_agents, H, W) float32  — per-agent ROI masks
                           in absolute board coordinates. Zero array for environments
                           without ROI components.
    }

Observations schema
-------------------
    {agent_id: {"vision": ndarray (C, H, W), "interoception": ndarray (N,), ...}}

Additional modalities may be added by concrete environments; consumers use only
what they declare in manifesto["observation_shapes"].

Manifesto schema (minimum)
--------------------------
    {
        "layers":             list[str],
        "observation_shapes": {modality_name: shape_tuple},
        "action_space":       list[str | int],
        "action_names":       {index: name},
        "food_ind":           int | None,
    }

Render manifest schema
----------------------
    {layer_name: keyword}

Maps each layer name (as it appears in state["layers"]) to a canonical renderer
keyword (e.g. "food" → "FOOD", "agent_0" → "AGENT_0"). The keyword vocabulary
is defined in renderer.py. Default is an empty dict — renders as blank floor.
Envs override to enable layout image output.
"""

from abc import ABC, abstractmethod


class AbstractEnv(ABC):
    @abstractmethod
    def reset(self, **kwargs):
        """Reset the environment.

        Returns:
            observations: {agent_id: observation_dict}
            state:        canonical world snapshot (see module docstring)
        """

    @abstractmethod
    def step_parallel(self, actions):
        """All agents act simultaneously.

        Args:
            actions: {agent_id: {"action": int, ...}}

        Returns:
            observations: {agent_id: observation_dict}
            state:        canonical world snapshot (see module docstring)
        """

    @abstractmethod
    def step_sequential(self, actions):
        """Agents act one at a time, observing intermediate effects.

        Not implemented in most environments; raises NotImplementedError.
        """

    @property
    @abstractmethod
    def manifesto(self):
        """Environment manifesto. Built on reset().

        Returns dict with at minimum the keys described in the module docstring.
        """

    @property
    @abstractmethod
    def score_dimensions(self):
        """List of score dimension names for event column registration.

        Returns [] for environments where reward inference is agent-side.
        """

    @property
    def render_manifest(self):
        """Layer name → renderer keyword mapping for layout image output.

        Keys must match strings that appear in state["layers"].
        Values are canonical renderer keywords defined in renderer.py.
        Default returns empty dict — renders as blank floor.
        Override to enable visible layout images.
        """
        return {}
