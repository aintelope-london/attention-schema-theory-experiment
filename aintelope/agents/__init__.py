# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Repository:
# https://github.com/aintelope-london/attention-schema-theory-experiment

from typing import Mapping, Type
from aintelope.agents.abstract_agent import AbstractAgent
from aintelope.agents.main_agent import MainAgent

# SB3 Discrete action space models
from aintelope.agents.ppo_agent import PPOAgent
from aintelope.agents.a2c_agent import A2CAgent
from aintelope.agents.dqn_agent import DQNAgent
from aintelope.agents.random_agent import RandomAgent
from aintelope.agents.llm_agent import LLMAgent

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

register_agent_class("sb3_ppo_agent", PPOAgent)
register_agent_class("sb3_dqn_agent", DQNAgent)
register_agent_class("sb3_a2c_agent", A2CAgent)

register_agent_class("llm_agent", LLMAgent)
