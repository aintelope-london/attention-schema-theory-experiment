# abstract_agent.py
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from abc import ABC, abstractmethod
from typing import Optional
from pathlib import Path


class AbstractAgent(ABC):
    @abstractmethod
    def reset(self, state, **kwargs) -> None:
        ...

    @abstractmethod
    def get_action(self, observation=None, **kwargs) -> Optional[dict]:
        """Returns a dict with at least {"action": int}. Additional keys
        (e.g. viewport masks, auxiliary outputs) are architecture-dependent
        and defined by the component connectome's output declarations."""
        ...

    @abstractmethod
    def update(self, observation=None, **kwargs) -> list:
        ...

    @abstractmethod
    def save_model(self, path: Path, **kwargs) -> None:
        ...
