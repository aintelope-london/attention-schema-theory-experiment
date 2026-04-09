# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from typing import Mapping, Type
from aintelope.agents.abstract_agent import AbstractAgent
from aintelope.agents.main_agent import MainAgent
from aintelope.agents.random_agent import RandomAgent
from aintelope.agents.sb3_agent import SB3Agent

AGENT_REGISTRY: Mapping[str, Type[AbstractAgent]] = {}


def register_agent_class(agent_id: str, agent_class: Type[AbstractAgent]):
    if agent_id in AGENT_REGISTRY:
        raise ValueError(f"{agent_id} is already registered")
    AGENT_REGISTRY[agent_id] = agent_class


def get_agent_class(agent_id: str) -> Type[AbstractAgent]:
    if agent_id not in AGENT_REGISTRY:
        raise ValueError(f"{agent_id} is not found in agent registry")
    return AGENT_REGISTRY[agent_id]


register_agent_class("random_agent", RandomAgent)
register_agent_class("main_agent", MainAgent)
register_agent_class("sb3_agent", SB3Agent)
