# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from aintelope.agents.abstract_agent import AbstractAgent
from aintelope.agents.model.model import Model
from aintelope.agents import scripts


class DummyAgent(AbstractAgent):
    """Scripted agent — a MainAgent variant with action-selection hijacked by
    an external script. The script plays the role MainAgent delegates to its
    Model's action component; the Model (always instantiated, empty by
    default) still runs its other components each step over the real
    observation, and surfaces their outputs in the returned action dict.

    Config entry:
        agent_0:
          agent_class: dummy_agent
          script: ROI_CONTRACTING       # name of sequence in scripts.py
          architecture:                 # optional — absent/null ⇒ no model
            roi: {type: ROI, ...}
            value: {type: StateValue-NN, inputs: [observation]}

    Use cases:
      - Animations: architecture declares ROI; its mask flows to the env.
      - Probes: architecture declares pre-trained predictors
        (value, dynamic, q_net, ...); each script step is one data point.
        Pass checkpoint= to load weights from a prior training block.

    Learning is disabled: update() is a no-op. Probe blocks run in test mode;
    Model components load frozen weights at instantiation time.

    When the script is exhausted the agent emits "wait" indefinitely.
    """

    def __init__(self, agent_id, env, cfg, checkpoint=None, **kwargs):
        self.id = agent_id
        self.last_action = 0
        agent_cfg = cfg.agent_params.agents[agent_id]
        self._sequence = list(getattr(scripts, agent_cfg.script))
        self._wait = env.manifesto["action_space"].index("wait")
        self._step = 0
        self.model = Model(agent_id, env.manifesto, cfg, checkpoint=checkpoint)

    def reset(self, observation, **kwargs):
        self._step = 0
        self.model.reset()

    def get_action(self, observation=None, **kwargs):
        entry = (
            self._sequence[self._step]
            if self._step < len(self._sequence)
            else {"action": self._wait}
        )
        self._step += 1
        self.last_action = entry["action"]

        activations = self.model.activations
        activations.update(observation)
        activations["internal_episode"] = self.model.resets
        activations["action"] = entry["action"]
        activations["internal_action"] = entry.get("internal_action", 1)

        output = dict(entry)
        for component_id, component in self.model.components.items():
            if component_id == "action":
                continue
            component.activate(activations)
            if component_id in activations:
                output[component_id] = activations[component_id]
        return output

    def update(self, observation=None, **kwargs):
        return {}

    def save_model(self, path, **kwargs):
        pass