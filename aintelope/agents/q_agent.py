from typing import Optional, Tuple, NamedTuple, List
import logging
import numpy.typing as npt

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import torch
from torch import nn

from gymnasium.spaces import Discrete

from aintelope.environments.savanna_gym import SavannaGymEnv #TODO used for hack
from aintelope.training.dqn_training import Trainer 
from aintelope.agents import (
    Agent,
    GymEnv,
    PettingZooEnv,
    Environment,
    register_agent_class,
)
from aintelope.agents.memory import Experience#, ReplayBuffer
from aintelope.environments.typing import (
    ObservationFloat,
    PositionFloat,
    Action,
    AgentId,
    AgentStates,
    Observation,
    Reward,
    Info,
)

logger = logging.getLogger("aintelope.agents.q_agent")


class HistoryStep(NamedTuple):
    state: NamedTuple
    action: int
    reward: float
    done: bool
    instinct_events: List[Tuple[str, int]]
    next_state: NamedTuple


class QAgent(Agent):
    """QAgent class, functioning as a base class for agents"""

    def __init__(
        self,
        # env: Environment,
        # model: nn.Module,
        # replay_buffer: ReplayBuffer,
        agent_id: str,
        trainer: Trainer,
        action_space: Discrete, 
        #warm_start_steps: int, # needs to go to experiment
        target_instincts: List[str] = [],
    ) -> None:
        # TODO: remove env calss from agent class
        # self.env = env
        self.id = agent_id
        self.action_space = action_space
        self.trainer = trainer
        #self.replay_buffer = replay_buffer
        # self.warm_start_steps = warm_start_steps
        self.history: List[HistoryStep] = []
        self.done = False
        self.last_action = 0
        #self.reset() #needs state, reset done in experiment.py where can ask for it

    def reset(self, state) -> None:
        """Resets self and updates the state."""
        self.done = False
        self.state = state
        if isinstance(self.state, tuple):
            self.state = self.state[0]

    def get_action(
        self, 
        observation: npt.NDArray[ObservationFloat] = None,
        step: int = 0, #net: nn.Module, epsilon: float, device: str
    ) -> Optional[int]:
        """Given an observation, ask your net what to do. State is needed to be given here
        as other agents have changed the state!

        Args:
            net: pytorch Module instance, the model
            epsilon: value to determine likelihood of taking a random action
            device: current device

        Returns:
            action (Optional[int]): index of action
        """
        if self.done:
            return None
        else:
            # observation goes to instincts here also
            action = self.trainer.get_action(self.id, observation, step) # TODO: better name, now same
        '''
        elif np.random.random() < epsilon:
            action = self.action_space.sample()
        else:
            logger.debug("debug state", type(state))
            state = torch.tensor(np.expand_dims(state, 0))
            logger.debug("debug state tensor", type(state), state.shape)
            if device not in ["cpu"]:
                state = state.cuda(device)

            q_values = net(state)
            _, action = torch.max(q_values, dim=1)
            action = int(action.item())
        '''
        self.last_action = action
        return action

    # double check that all the logic of `play_step` is within experiments.py and then
    # delete this method
    @torch.no_grad()
    def play_step(
        self,
        net: nn.Module,
        epsilon: float = 0.0,
        device: str = "cpu",
        save_path: Optional[str] = None,
    ) -> Tuple[float, float, bool]:
        """
        Only for Gym envs, not PettingZoo envs
        Carries out a single interaction step between the agent and the
        environment.

        Args:
            net: DQN network
            epsilon: value to determine likelihood of taking a random action
            device: current device

        Returns:
            reward, score, done
        """

        '''
        # The 'mind' (model) of the agent decides what to do next
        action = self.get_action(net, epsilon, device)

        # do step in the environment
        # the environment reports the result of that decision
        if isinstance(self.env, GymEnv):
            next_state, env_reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
        elif isinstance(self.env, PettingZooEnv):
            next_state, env_reward, terminateds, truncateds, _ = self.env.step(action)
            done = {
                key: terminated or truncateds[key]
                for (key, terminated) in terminateds.items()
            }
        else:
            logger.warning(f"{self.env} is not of type GymEnv or PettingZooEnv")
            next_state, env_reward, done, _ = self.env.step(action)
        '''
    def update(
        self,
        env: SavannaGymEnv = None, #TODO hack, figure out if state_to_namedtuple can be static somewhere
        observation: npt.NDArray[ObservationFloat] = None,
        score: float = 0.0,
        done: bool = False,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Takes observations and updates trainer on perceived experiences. Needed here to catch instincts.

        Args:
            observation: ObservationArray
            score: Only baseline uses score as a reward
            done: boolean whether run is done 

        Returns:
            None
        """
        next_state = observation # add state (interoception) handling here when needed
        exp = Experience(self.state, self.last_action, score, done, next_state)
        self.history.append(
            HistoryStep(
                state=env.state_to_namedtuple(self.state.tolist()),
                action=self.last_action,
                reward=score,
                done=done,
                instinct_events=[],
                next_state=env.state_to_namedtuple(next_state.tolist()),
            )
        )

        if save_path is not None:
            with open(save_path, "a+") as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow(
                    [
                        self.state.tolist(),
                        action,
                        score,
                        done,
                        [],
                        next_state,
                    ]
                )

        #self.replay_buffer.append(exp)
        self.trainer.update_memory(self.id, exp)
        self.state = next_state

        # if scenario is complete or agent experiences catastrophic failure,
        # end the agent.
        if done:
            self.reset()

    def get_history(self) -> pd.DataFrame:
        """
        Method to get the history of the agent. Note that warm_start_steps are excluded.
        """
        return pd.DataFrame(
            columns=[
                "state",
                "action",
                "reward",
                "done",
                "instinct_events",
                "next_state",
            ],
            data=self.history[self.warm_start_steps :],
        )

    @staticmethod
    def process_history(
        history_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Function to convert the agent history dataframe into individual dataframe for
        agent position, grass and water locations. Instinct events are currently not
        processed.
        """
        state_df = pd.DataFrame(history_df.state.to_list())
        agent_df = pd.DataFrame(
            columns=["x", "y"], data=state_df.agent_coords.to_list()
        )
        grass_columns = [c for c in list(state_df) if c.startswith("grass")]
        grass_df = state_df[grass_columns].applymap(lambda x: tuple(x))
        grass_df = pd.DataFrame(
            columns=["x", "y"], data=set(grass_df.stack().to_list())
        )
        water_columns = [c for c in list(state_df) if c.startswith("water")]
        water_df = state_df[water_columns].applymap(lambda x: tuple(x))
        water_df = pd.DataFrame(
            columns=["x", "y"], data=set(water_df.stack().to_list())
        )

        return agent_df, grass_df, water_df

    def plot_history(self, style: str = "thickness", color: str = "viridis") -> Figure:
        history_df = self.get_history()
        agent_df, food_df, water_df = self.process_history(history_df)

        fig, ax = plt.subplots(figsize=(8, 8))

        if style == "thickness":
            ax.plot(agent_df["x"], agent_df["y"], ".r-")
        elif style == "colormap":
            cmap = matplotlib.colormaps[color]

            agent_arr = agent_df.to_numpy()  # coordinates x y
            # coordinates are ordered in x1 y1 x2 y2
            step_pairs = np.concatenate([agent_arr[:-1], agent_arr[1:]], axis=1)
            unique_steps, step_freq = np.unique(step_pairs, axis=0, return_counts=True)

            for line_segment, col in zip(unique_steps, step_freq / step_freq.max()):
                if (line_segment[:2] == line_segment[2:]).all():  # agent did not move
                    im = ax.scatter(
                        line_segment[0],
                        line_segment[1],
                        s=70,
                        marker="o",
                        color=cmap(col),
                    )
                else:
                    ax.plot(line_segment[[0, 2]], line_segment[[1, 3]], color=cmap(col))

            cbar = fig.colorbar(im)
            cbar.set_label("Relative Frequency Agent")
        else:
            raise NotImplementedError(f"{style} is not a valid plot style!")

        ax.plot(food_df["x"], food_df["y"], "xg", markersize=8, label="Food")
        ax.plot(water_df["x"], water_df["y"], "xb", markersize=8, label="Water")
        ax.legend()
        plt.tight_layout()
        return fig


register_agent_class("q_agent", QAgent)
