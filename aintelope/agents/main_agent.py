# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Repository: https://github.com/aintelope-london/attention-schema-theory-experiment

from typing import Optional

from aintelope.agents.abstract_agent import AbstractAgent
from aintelope.agents.model.model import Model


class MainAgent(AbstractAgent):
    """Canon agent."""

    def __init__(self, agent_id, env=None, cfg=None, checkpoint=None, **kwargs):
        self.id = agent_id
        self.cfg = cfg
        self.done = False
        self.last_action = None
        self.model = Model(agent_id, env.manifesto, cfg, checkpoint=checkpoint)

    def reset(self, state, **kwargs):
        self.done = False
        self.last_action = None
        self.observation = state

    def get_action(self, observation=None, **kwargs) -> Optional[int]:
        if self.done:
            return None
        self.observation = observation
        action = self.model.get_action(observation)
        self.last_action = action
        return action

    def update(self, observation=None, **kwargs):
        return self.model.update(observation)

    def save_model(self, path, **kwargs):
        self.model.save(path)
