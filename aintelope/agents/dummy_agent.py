# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np
from aintelope.agents.abstract_agent import AbstractAgent
from aintelope.agents.model.roi import ROI
from aintelope.agents import scripts


class DummyAgent(AbstractAgent):
    """Scripted agent — replays a named action sequence from agents/scripts.py.

    Config entry:
        agent_0:
          agent_class: dummy_agent
          script: ROI_CONTRACTING       # name of sequence in scripts.py
          architecture:                 # optional — include only if ROI overlay needed
            roi:
              type: ROI
              ...

    Each entry in the script is {"action": int} plus optionally
    {"internal_action": int}. When an architecture with a ROI entry is
    declared, the ROI component is instantiated and driven by internal_action,
    and its mask is appended to the output as "roi".

    When the sequence is exhausted, the agent emits wait (action index for
    "wait" in the manifesto) indefinitely.
    """

    def __init__(self, agent_id, env, cfg, **kwargs):
        self.id = agent_id
        self.last_action = 0
        agent_cfg = cfg.agent_params.agents[agent_id]
        self._sequence = list(getattr(scripts, agent_cfg.script))
        self._wait = env.manifesto["action_space"].index("wait")
        self._step = 0
        self._roi = self._init_roi(agent_cfg)
        self._vision_shape = env.manifesto["observation_shapes"]["vision"]

    def _init_roi(self, agent_cfg):
        architecture = agent_cfg.get("architecture", {})
        roi_entry = {k: v for k, v in architecture.items() if v.get("type") == "ROI"}
        if not roi_entry:
            return None
        component_id, plans = next(iter(roi_entry.items()))
        return ROI(
            {
                "plans": plans,
                "component_id": component_id,
                "inputs": ["internal_action"],
                "components": {},
                "activations": {},
                "cfg": {},
            }
        )

    def reset(self, observation, **kwargs):
        self._step = 0
        if self._roi:
            self._roi.reset()

    def get_action(self, observation=None, **kwargs):
        entry = (
            self._sequence[self._step]
            if self._step < len(self._sequence)
            else {"action": self._wait}
        )
        self._step += 1
        self.last_action = entry["action"]

        if self._roi is None:
            return dict(entry)

        vision = np.ones(self._vision_shape, dtype=np.float32)
        activations = {
            "vision": vision,
            "internal_action": entry.get("internal_action", 1),
        }
        self._roi.activate(activations)
        return {**entry, "roi": activations[self._roi.component_id]}

    def update(self, observation=None, **kwargs):
        return {}

    def save_model(self, path, **kwargs):
        pass
