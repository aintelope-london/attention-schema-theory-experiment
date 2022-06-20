import typing as typ
import functools

import numpy as np
from gym.spaces import Discrete, Box
from pettingzoo import AECEnv
from pettingzoo.utils import wrappers, agent_selector

NUM_ITERS = 500  # duration of the game
MAP_MIN = 0
MAP_DIM = 100
CYCLIC_BOUNDARIES = True
AMOUNT_AGENTS = 1  # for now only one agent
AMOUNT_GRASS = 2
Float = np.float32
OBSERVATION_SPACE = Box(0,
                        MAP_DIM,
                        shape=(2 * (AMOUNT_AGENTS + AMOUNT_GRASS), ))
ACTION_SPACE = Discrete(4)  # agent can walk in 4 directions
ACTION_MAP = np.array([[0, 1], [1, 0], [0, -1], [-1, 0]], dtype=Float)


class RawEnv(AECEnv):

    metadata = {'name': 'savanna_v1'}

    def __init__(self):
        self.possible_agents = [
            'player_' + str(r) for r in range(AMOUNT_AGENTS)
        ]
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(AMOUNT_AGENTS))))

        self._action_spaces = {
            agent: ACTION_SPACE
            for agent in self.possible_agents
        }
        self._observation_spaces = {
            agent: OBSERVATION_SPACE
            for agent in self.possible_agents
        }

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str):
        return OBSERVATION_SPACE

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str):
        return ACTION_SPACE

    def observe(self, agent: str):
        """Return observation of given agent.
        """
        return np.array(self.observations[agent])

    def render(self, mode='human'):
        """Render the environment.
        """
        raise NotImplementedError

    def close(self):
        """Release any graphical display, subprocesses, network connections
        or any other environment data which should not be kept around after
        the user is no longer using the environment.
        """
        raise NotImplementedError

    def reset(self, seed: typ.Optional[int] = None):
        """Reset needs to initialize the following attributes:
            - agents
            - rewards
            - _cumulative_rewards
            - dones
            - infos
            - agent_selection
        And must set up the environment so that render(), step(), and observe()
        can be called without issues.
        """

        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.grass = np.random.randint(0, MAP_DIM, 2 * AMOUNT_GRASS)
        self.state = {
            agent: np.array(np.random.randint(0, MAP_DIM, 2), dtype=Float)
            for agent in self.agents
        }
        self.observations = {
            agent: np.concatenate([self.state[agent], self.grass])
            for agent in self.agents
        }
        self.num_moves = 0

        # cycle through the agents
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

    def step(self, action: int):
        """Take in an action for the current agent (specified by
        agent_selection) and needs to update:
        - rewards
        - _cumulative_rewards
        - dones
        - infos
        - agent_selection (to the next agent)
        And any internal state used by observe() or render()
        """
        if self.dones[self.agent_selection]:
            """
            handles stepping an agent which is already done
            accepts a None action for the one agent, and moves the
            agent_selection to the next done agent, or if there are no more
            done agents, to the next live agent
            """
            return self._was_done_step(action)

        agent = self.agent_selection
        """
        the agent which stepped last had its _cumulative_rewards accounted for
        (because it was returned by last()), so the _cumulative_rewards for
        this agent should start again at 0
        """
        self._cumulative_rewards[agent] = 0

        move = ACTION_MAP[action]
        # stores action of current agent
        agent_pos = self.state[self.agent_selection]
        agent_pos += move
        agent_pos = np.clip(agent_pos, MAP_MIN, MAP_DIM)
        self.state[self.agent_selection] = agent_pos


        # collect reward if it is the last agent to act
        if self._agent_selector.is_last():
            # rewards for all agents are placed in the .rewards dictionary
            for iagent, agent in enumerate(self.agents):
                agent_pos = self.state[agent]
                def distance(a, b):
                    return np.linalg.norm(np.vstack((a, b)))
                reward = min(distance(agent_pos, grass_pos) for grass_pos in self.grass)
                self.rewards[agent] = reward

            self.num_moves += 1
            # The dones dictionary must be updated for all players.
            self.dones = {
                agent: self.num_moves >= NUM_ITERS
                for agent in self.agents
            }

            # observe the current state
            for agent in self.agents:
                iagent = self.agent_name_mapping[agent]
                self.observations[iagent] = self.state[agent]
        else:
            # necessary so that observe() returns a reasonable observation at all times.
            iagent = 1 - self.agent_name_mapping[agent]
            self.state[self.agents[iagent]] = None
            # no rewards are allocated until both players give an action
            self._clear_rewards()

        # selects the next agent.
        self.agent_selection = self._agent_selector.next()
        # Adds .rewards to ._cumulative_rewards
        self._accumulate_rewards()


def env():
    """Add PettingZoo wrappers to environment class.
    """
    env = RawEnv()
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


def main(env: RawEnv):
    env.reset()
    for agent in env.agent_iter():
        observation, reward, done, info = env.last()

        action_space:Discrete = env.action_space(agent)


        # action = policy(observation, agent)
        if not done:
            action = action_space.sample()
        else:
            action = None
        env.step(action)
        env.render('human')


if __name__ == '__main__':
    e = env()
    main(e)
