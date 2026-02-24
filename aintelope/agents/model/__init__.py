# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Repository: https://github.com/aintelope/biological-compatibility-benchmarks


# NOT USED ATM, TODO, consider if we'd rather use the globals() or not everywhere
"""
from typing import Mapping, Type
from aintelope.agents.model import dl_components
from aintelope.agents.model import affects

COMPONENT_REGISTRY: Mapping[str, Type[AbstractAgent]] = {}


def register_agent_class(component_id: str, agent_class: Type[AbstractAgent]):
    if component_id in COMPONENT_REGISTRY:
        raise ValueError(f"{component_id} is already registered")
    COMPONENT_REGISTRY[component_id] = agent_class


def get_agent_class(component_id: str) -> Type[AbstractAgent]:
    if component_id not in COMPONENT_REGISTRY:
        raise ValueError(f"{component_id} is not found in agent registry")
    return COMPONENT_REGISTRY[component_id]


# add component class to registry
register_agent_class("random_agent", RandomAgent)
register_agent_class("affect_agent", Agent)
register_agent_class("baby_agent", BabyAgent)
"""
