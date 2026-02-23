from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any


class Component(ABC):
    """
    Lightweight abstract mixin for model components.
    - No __init__ to avoid nn.Module / MRO initializer conflicts (NeuralNet can inherit as nn.Module, Component).
    - activate(...) returns (outputs_dict, confidence: float).
    - update() performs maintenance / learning and returns a report (dict or None).
    """

    @abstractmethod
    def activate(self, observation: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
        raise NotImplementedError

    @abstractmethod
    def update(self) -> Any:
        raise NotImplementedError
