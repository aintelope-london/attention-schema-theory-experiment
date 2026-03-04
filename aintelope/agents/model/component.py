from abc import ABC, abstractmethod
from typing import Dict, Any


class Component(ABC):
    """
    Abstract mixin for connectome components.
    - No __init__ to avoid nn.Module / MRO conflicts.
    - activate() writes to the shared activations dict.
    - update() performs learning and returns a report (dict or None).
    """

    @abstractmethod
    def activate(self, activations: Dict[str, Any]) -> None:
        raise NotImplementedError

    @abstractmethod
    def update(self) -> Any:
        raise NotImplementedError
