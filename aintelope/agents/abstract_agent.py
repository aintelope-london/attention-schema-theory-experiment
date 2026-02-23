# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Repository:
# https://github.com/aintelope-london/attention-schema-theory-experiment

from abc import ABC, abstractmethod
from typing import Optional
from pathlib import Path


class AbstractAgent(ABC):
    @abstractmethod
    def reset(self, state, **kwargs) -> None:
        ...

    @abstractmethod
    def get_action(self, observation=None, **kwargs) -> Optional[int]:
        ...

    @abstractmethod
    def update(self, observation=None, **kwargs) -> list:
        ...

    @abstractmethod
    def save_model(self, path: Path, **kwargs) -> None:
        ...
