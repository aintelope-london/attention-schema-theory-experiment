# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Repository:
# https://github.com/biological-alignment-benchmarks/biological-alignment-gridworlds-benchmarks

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union
import numpy.typing as npt
from pathlib import Path

from aintelope.aintelope_typing import ObservationFloat
from pettingzoo import AECEnv, ParallelEnv

Environment = Union[AECEnv, ParallelEnv]


class Agent(ABC):
    @abstractmethod
    def reset(self, state, info, env_class) -> None:
        ...

    @abstractmethod
    def get_action(
        self,
        observation: Tuple[
            npt.NDArray[ObservationFloat], npt.NDArray[ObservationFloat]
        ] = None,
        info: dict = {},
        step: int = 0,
        env_layout_seed: int = 0,
        episode: int = 0,
        trial: int = 0,
        *args,
        **kwargs,
    ) -> Optional[int]:
        ...

    @abstractmethod
    def update(
        self,
        env: Environment = None,
        observation: Tuple[
            npt.NDArray[ObservationFloat], npt.NDArray[ObservationFloat]
        ] = None,
        info: dict = {},
        score: float = 0.0,
        done: bool = False,
    ) -> list:
        ...

    @abstractmethod
    def init_model(
        self,
        observation_shape,
        action_space,
        checkpoint: Optional[Path] = None,
    ) -> None:
        ...

    def save_model(
        self,
        path: Path,
    ) -> None:
        self.trainer.save_model(self.id, path)
