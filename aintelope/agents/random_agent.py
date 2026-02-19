from aintelope.agents.abstract_agent import Agent


class RandomAgent(Agent):
    def __init__(self, agent_id, trainer=None, env=None, cfg=None, **kwargs):
        self.id = agent_id
        self.done = False

    def reset(self, state, info, env_class):
        self.done = False

    def init_model(self, observation_shape, action_space, checkpoint=None):
        self.action_space = action_space

    def get_action(self, *args, **kwargs):
        self.last_action = self.action_space.sample()
        return self.last_action

    def update(self, observation=None, score=0.0, done=False, **kwargs):
        self.state = observation
        event = [self.id, self.state, self.last_action, score, done, observation]
        return event

    def save_model(self, path):
        pass
