# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Repository:
# https://github.com/aintelope-london/attention-schema-theory-experiment

from typing import Mapping, Type

from aintelope.environments.abstract_env import AbstractEnv
from aintelope.environments.savanna_wrapper import SavannaWrapper

ENV_REGISTRY: Mapping[str, Type[AbstractEnv]] = {}


def register_env_class(env_id: str, env_class: Type[AbstractEnv]):
    if env_id in ENV_REGISTRY:
        raise ValueError(f"{env_id} is already registered")
    ENV_REGISTRY[env_id] = env_class


def get_env_class(env_id: str) -> Type[AbstractEnv]:
    if env_id not in ENV_REGISTRY:
        raise ValueError(f"{env_id} is not found in env registry")
    return ENV_REGISTRY[env_id]


register_env_class("savanna-safetygrid-v1", SavannaWrapper)
