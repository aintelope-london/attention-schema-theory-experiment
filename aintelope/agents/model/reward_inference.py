import numpy as np
import math
from aintelope.agents.model.component import Component


class RewardInference(Component):
    """
    Infers reward from observation dict.
    Environment-specific reward logic — hand-crafted for savanna gridworld.
    Registered as the 'reward' role in the agent's architecture config.
    """

    def __init__(self, context):
        self.cfg = context["cfg"]
        self.env_manifesto = context["env_manifesto"]
        self.role = context["role"]
        self.metadata = context["plans"]["metadata"]
        self.satiation = 0

    def activate(self, observation):
        reward = 0.0

        interoception = observation.get("interoception", np.array([]))
        vision = observation.get("vision", None)

        # Food reward — homeostatic
        if len(interoception) > 0:
            food_signal = interoception[0]
            if food_signal > 0:
                reward += food_signal - self.satiation
                self.satiation = min(self.satiation + 10, 10)

        # Proximity reward — smell
        if vision is not None and "food_ind" in self.env_manifesto:
            food_ind = self.env_manifesto["food_ind"]
            agent_center = [vision.shape[1] / 2, vision.shape[2] / 2]
            food_coords = tuple(zip(*np.where(vision[food_ind] > 0)))
            if food_coords:
                closest = min(
                    math.sqrt(
                        (fc[0] - agent_center[0]) ** 2 + (fc[1] - agent_center[1]) ** 2
                    )
                    for fc in food_coords
                )
                reward += 10 - closest

        # Homeostasis — inverted U
        if self.satiation > -10:
            self.satiation -= 1
        reward += -abs(self.satiation)

        return {"reward": np.atleast_1d(np.float32(reward))}, 1.0

    def update(self):
        return None
