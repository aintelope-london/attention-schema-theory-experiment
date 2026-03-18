# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np
import math
from abc import ABC, abstractmethod
from aintelope.agents.model.component import Component


# ── Reward scheme ABC ─────────────────────────────────────────────────


class RewardScheme(ABC):
    """Base class for all reward schemes.

    Each scheme encapsulates one reward signal. Stateless schemes
    inherit the no-op reset. Stateful schemes (e.g. Homeostasis)
    override reset() to clear their internal state.
    """

    @abstractmethod
    def activate(self, activations: dict, env_manifesto: dict) -> float:
        raise NotImplementedError

    def reset(self) -> None:
        pass


# ── Active reward schemes ─────────────────────────────────────────────


class FoodInteroception(RewardScheme):
    """Reward from food interoception signal.

    Reads interoception[0] directly as the reward value.
    Positive when the agent is eating, zero otherwise. No state.
    """

    def activate(self, activations: dict, env_manifesto: dict) -> float:
        interoception = activations.get("next_interoception", activations.get("interoception", np.array([])))
        if len(interoception) > 0 and interoception[0] > 0:
            return float(interoception[0])
        return 0.0


# ── WIP reward schemes ────────────────────────────────────────────────
# WIP: not tested, not activated. Do not add to metadata.rewards yet.


class FoodSmell(RewardScheme):
    """WIP: Proximity reward based on distance to nearest food in vision.

    Rewards the agent for being close to food, encouraging approach behaviour.
    Requires food_ind to be present in env_manifesto.
    """

    def activate(self, activations: dict, env_manifesto: dict) -> float:
        vision = activations.get("vision", None)
        if vision is None or "food_ind" not in env_manifesto:
            return 0.0
        food_ind = env_manifesto["food_ind"]
        if food_ind is None:
            return 0.0
        agent_center = [vision.shape[1] / 2, vision.shape[2] / 2]
        food_coords = tuple(zip(*np.where(vision[food_ind] > 0)))
        if not food_coords:
            return 0.0
        closest = min(
            math.sqrt((fc[0] - agent_center[0]) ** 2 + (fc[1] - agent_center[1]) ** 2)
            for fc in food_coords
        )
        return 10.0 - closest


class Homeostasis(RewardScheme):
    """WIP: Inverted-U homeostatic reward with satiation state.

    Penalises both over- and under-satiation. Satiation rises on eating
    and decays each step. Requires FoodInteroception to be active too,
    or equivalent eating detection logic.
    """

    def __init__(self):
        self.satiation = 0.0

    def activate(self, activations: dict, env_manifesto: dict) -> float:
        interoception = activations.get("next_interoception", activations.get("interoception", np.array([])))
        if len(interoception) > 0 and interoception[0] > 0:
            self.satiation = min(self.satiation + 10.0, 10.0)
        if self.satiation > -10.0:
            self.satiation -= 1.0
        return -abs(self.satiation)

    def reset(self) -> None:
        self.satiation = 0.0


# ── Registry ──────────────────────────────────────────────────────────

_REWARD_REGISTRY = {
    "FoodInteroception": FoodInteroception,
    "FoodSmell": FoodSmell,
    "Homeostasis": Homeostasis,
}


# ── Strategy component ────────────────────────────────────────────────


class RewardInference(Component):
    """Infers reward from observation by delegating to reward scheme instances.

    Active schemes are declared in metadata.rewards config list.
    Each scheme is instantiated at init, called in sequence during activate,
    and reset at episode boundaries.
    """

    def __init__(self, context):
        self.component_id = context["component_id"]
        self.env_manifesto = context["env_manifesto"]
        self.metadata = context["plans"]["metadata"]

        reward_names = self.metadata.get("rewards", ["FoodInteroception"])
        self.rewards = []
        for name in reward_names:
            if name not in _REWARD_REGISTRY:
                raise ValueError(
                    f"RewardInference: unknown reward scheme '{name}'. "
                    f"Available: {list(_REWARD_REGISTRY.keys())}"
                )
            self.rewards.append(_REWARD_REGISTRY[name]())

    def activate(self, activations: dict) -> None:
        total = sum(
            scheme.activate(activations, self.env_manifesto) for scheme in self.rewards
        )
        activations[self.component_id] = np.atleast_1d(np.float32(total))

    def reset(self) -> None:
        for scheme in self.rewards:
            scheme.reset()

    def update(self, signals=None):
        return None
