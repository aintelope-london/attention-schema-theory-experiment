from aintelope.agents.abstract_agent import Agent


class RandomAgent(Agent):
    def reset(self, state, info, env_class):
        self.done = False

    def init_model(self, observation_shape, action_space, checkpoint=None):
        self.action_space = action_space

    def get_action(self, *args, **kwargs):
        return self.action_space.sample()

    def update(self, **kwargs):
        return []
