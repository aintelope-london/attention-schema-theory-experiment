# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import random
from aintelope.agents.abstract_agent import AbstractAgent


class RandomAgent(AbstractAgent):
    def __init__(self, agent_id, env=None, cfg=None, **kwargs):
        self.id = agent_id
        self.done = False
        self.last_action = None
        self.action_space = env.manifesto["action_space"]

    def reset(self, state, **kwargs):
        self.done = False
        self.last_action = None

    def get_action(self, observation=None, **kwargs):
        self.last_action = random.choice(self.action_space)
        return {"action": self.last_action}

    def update(self, observation=None, **kwargs):
        return {}

    def save_model(self, path, **kwargs):
        pass
