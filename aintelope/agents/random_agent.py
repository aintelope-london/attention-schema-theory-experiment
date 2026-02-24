import random
from aintelope.agents.abstract_agent import AbstractAgent


class RandomAgent(AbstractAgent):
    def __init__(self, agent_id, env=None, cfg=None, **kwargs):
        self.id = agent_id
        self.done = False
        self.last_action = None
        self.action_space = env.manifesto["action_space"]

    def reset(self, state, **kwargs):
        self.done = False
        self.last_action = None

    def get_action(self, observation=None, **kwargs):
        self.last_action = random.choice(self.action_space)
        return self.last_action

    def update(self, observation=None, **kwargs):
        return []

    def save_model(self, path, **kwargs):
        pass
