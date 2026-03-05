# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from abc import ABC, abstractmethod
from typing import Dict, Any


class Component(ABC):
    """
    Abstract mixin for connectome components.
    - No __init__ to avoid nn.Module / MRO conflicts.
    - activate() writes to the shared activations dict.
    - update() performs learning and returns a report (dict or None).
    - reset() is called at episode boundaries. Default is no-op.
    """

    @abstractmethod
    def activate(self, activations: Dict[str, Any]) -> None:
        raise NotImplementedError

    @abstractmethod
    def update(self, signals=None) -> Any:
        raise NotImplementedError

    def reset(self) -> None:
        """Episode boundary reset. Override in stateful components."""
        pass