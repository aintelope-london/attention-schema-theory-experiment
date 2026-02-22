# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Repository: https://github.com/aintelope/biological-compatibility-benchmarks

from typing import Optional

from omegaconf import DictConfig

from aintelope.agents.abstract_agent import AbstractAgent
from aintelope.agents.model.trainer import Trainer

class MainAgent(AbstractAgent):
    """Main agent class"""

    trainer = None

    def __init__(
        self,
        agent_id: str,
        cfg: DictConfig,
        env_manifesto: dict,
    ) -> None:
        self.id = agent_id
        self.cfg = cfg
        self.done = False
        self.last_action = None
        #self.actions = env_manifesto["action_space"]
        #self.action_to_idx = {name: idx for idx, name in enumerate(self.actions)}

        if MainAgent.trainer is None:
            MainAgent.trainer = Trainer(cfg)
        MainAgent.trainer.add_agent(agent_id, env_manifesto)

        #self.affects = Affects(cfg, env_manifesto) 

    def reset(self, state) -> None:
        """Resets self and updates the state."""
        self.done = False
        self.last_action = None
        self.observation = state

    def get_action(self, observation) -> Optional[str]:
        """Observation coming from update
        Returns:
            action (Optional[int]): index of action
        """

        if self.done:
            return None

        self.observation = observation
        #action_idx = Agent.trainer.get_action(self.id, self.observation)
        #action = self.actions[action_idx]
        #confidence = 1.0
        #action, reward = self.affects.activate(self.observation, confidence, action) #
        action = MainAgent.trainer.get_action(self.id, self.observation)
        #self.reward = reward
        self.last_action = action
        return {"Agent_id": self.id, "Action": action}#, "Reward": self.reward}

    def update(
        self,
        next_observation=None,
        done=False,
    ) -> list:
        '''
        event = [
            self.id,
            self.observation,
            self.last_action,
            self.reward,
            self.done,
            next_observation,
        ]
        '''
        self.done = done
        #if not self.cfg.run_params.test_mode:
        #    Agent.trainer.update(*event)
        event = MainAgent.trainer.update(next_observation, self.done)
        return event

    def save_models(self, episode):
        MainAgent.trainer.save_models(episode)
