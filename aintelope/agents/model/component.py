from abc import ABC, abstractmethod
from typing import Dict, Any


class Component(ABC):
    @abstractmethod
    def activate(self, activations: Dict[str, Any]) -> None:
        raise NotImplementedError

    @abstractmethod
    def update(self, signals: Dict[str, Any]) -> Any:
        raise NotImplementedError
