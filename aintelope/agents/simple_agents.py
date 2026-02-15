# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Repository:
# https://github.com/biological-alignment-benchmarks/biological-alignment-gridworlds-benchmarks

"""
Collection of simple (rule based) agents, e.g. a completely random agent
"""

import logging

from aintelope.agents.q_agent import QAgent


class RandomAgent(QAgent):
    def get_action(self, *args, **kwargs) -> int:
        action_space = self.trainer.action_spaces[self.id]
        return action_space.sample()
