import numpy as np
import math


class RewardInference:
    def __init__(self, env_manifesto):
        self.satiation = 0
        self.manifesto = env_manifesto

    def infer_reward(self, state, action, next_state, info):
        """
        Example on how to infer the reward from the observation in example_agent.py.
        Use the environments allowed definitions to make it easier within the allowed
        limits. You can't "deduce what a food would look like" in our environment after
        all.
        The example contains a simple homeostatically charged reward from eating.
        The hungried the agent was, the more reward it will get from eating.
        If it was satiated though, it will get a negative reward instead from overeating.
        Action and next_state are allowed for your convenience, but not used in this
        example.

        Parameters:
            state: Observation
            action: int
            next_state: Observation
        Returns:
            Reward: float
        """

        reward = 0.0

        # Food reward
        interoception_food = state[1][0]

        if interoception_food > 0:
            reward += interoception_food - self.satiation
            self.satiation += 10
            if self.satiation > 10:
                self.satiation = 10

        # Smell reward (remove from example)
        agent_coordinates = [state[0].shape[0] / 2, state[0].shape[1] / 2]
        food_ind = self.manifesto["food_ind"]
        all_food_coordinates = tuple(zip(*np.where(state[0][food_ind] > 0)))
        if len(all_food_coordinates) > 0:
            closest_food = math.inf
            for food_coordinates in all_food_coordinates:
                xd = food_coordinates[0] - agent_coordinates[0]
                yd = food_coordinates[1] - agent_coordinates[1]
                distance = math.sqrt(xd * xd + yd * yd)
                if distance < closest_food:
                    closest_food = distance

            reward += 10 - closest_food

        # Homeostasis reward (inverted U -function)
        if self.satiation > -10:
            self.satiation -= 1
        reward += -abs(self.satiation)

        return reward
