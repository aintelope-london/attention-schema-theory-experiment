from typing import Dict, List, Optional, NamedTuple, Tuple
import typing as typ
import logging
from pathlib import Path
from collections import OrderedDict

from omegaconf import DictConfig
import numpy.typing as npt
import numpy as np
import gymnasium as gym

import torch
from torch import Tensor, nn
#from torch.optim import Adam, Optimizer
import torch.optim as optim
#from torch.utils.data import DataLoader
#from pytorch_lightning import LightningModule, Trainer, loggers as pl_loggers
#from pytorch_lightning.utilities.enums import DistributedType
#from pytorch_lightning.callbacks import ModelCheckpoint

from aintelope.agents.memory import ReplayBuffer#, RLDataset
from aintelope.agents import get_agent_class
from aintelope.models.dqn import DQN
from aintelope.environments.savanna_gym import SavannaGymEnv
from aintelope.agents.memory import Experience, ReplayBuffer

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

class Trainer:
    
    def __init__(self, params, n_observations, action_space):
        self.policy_nets = {}
        self.target_nets = {}
        self.replay_buffers = {}
        
        self.n_observations = n_observations
        self.action_space = action_space # TODO add these to the config? 
        self.hparams = params.hparams # now have to give whole cfg, trainer/hparams...
        '''
        BATCH_SIZE = 128
        GAMMA = 0.99
        EPS_START = 0.9
        EPS_END = 0.05
        EPS_DECAY = 1000
        '''
        self.TAU = 0.005
        LR = 1e-4
        # tb-logging and device control, check lightning_Trainer for 'AVAIL'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.optimizer = optim.AdamW(
            DQN(self.n_observations, self.action_space.n).parameters(), lr=LR, amsgrad=True
        ) #refactor, making a dummy network now. problem is add_agent inits first ---v

    def add_agent(self, agent_id):
        self.replay_buffers[agent_id] = ReplayBuffer(self.hparams.replay_size)
        self.policy_nets[agent_id] = DQN(self.n_observations, self.action_space.n).to(self.device)
        self.target_nets[agent_id] = DQN(self.n_observations, self.action_space.n).to(self.device)
        self.target_nets[agent_id].load_state_dict(self.policy_nets[agent_id].state_dict())

    @torch.no_grad() # TODO this might not be in the right place!
    def get_action(self, 
                   agent_id: str = "",
                   observation: npt.NDArray[ObservationFloat] = None, 
                   step: int = 0
                  )-> Optional[int]:
        
        epsilon = max(
                self.hparams.eps_end,
                self.hparams.eps_start - step * 1 / self.hparams.eps_last_frame,
            )
        if np.random.random() < epsilon:
            action = self.action_space.sample()
        else:
            logger.debug("debug state", type(state))
            state = torch.tensor(np.expand_dims(state, 0))
            logger.debug("debug state tensor", type(state), state.shape)
            if self.device not in ["cpu"]:
                state = state.cuda(self.device)

            q_values = net(state)
            _, action = torch.max(q_values, dim=1)
            action = int(action.item())
        
        return action
    
    
    def update_memory(self, agent_id: str, exp: Experience):
        self.replay_buffers[agent_id].append(exp)
        
    # replace by experience
    #Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))

    #TODO: is optimizer supposed to be done one step at a time?
    def optimize_models(self, step): #optimizer, memory, policy_net, target_net):
        for agent_id in self.policy_nets.keys():
            
            #if len(memory) < BATCH_SIZE:
            if len(self.replay_buffers[agent_id]) < self.hparams.batch_size:
                return
            
            #transitions = memory.sample(BATCH_SIZE)
            # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
            # detailed explanation). This converts batch-array of Transitions
            # to Transition of batch-arrays.
            #batch = Transition(*zip(*transitions))
            transitions = self.replay_buffers[agent_id].sample(self.hparams.batch_size)
            batch = Transition(*zip(*transitions))
            
            # Compute a mask of non-final states and concatenate the batch elements
            # (a final state would've been the one after which simulation ended)
            non_final_mask = torch.tensor(
                tuple(map(lambda s: s is not None, batch.next_state)),
                device=self.device,
                dtype=torch.bool,
            )
            non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
            state_batch = torch.cat(batch.state)
            action_batch = torch.cat(batch.action)
            reward_batch = torch.cat(batch.reward)

            # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
            # columns of actions taken. These are the actions which would've been taken
            # for each batch state according to policy_net
            policy_net = self.policy_nets[agent_id]
            target_net = self.target_nets[agent_id]
            state_action_values = policy_net(state_batch).gather(1, action_batch)

            # Compute V(s_{t+1}) for all next states.
            # Expected values of actions for non_final_next_states are computed based
            # on the "older" target_net; selecting their best reward with max(1)[0].
            # This is merged based on the mask, such that we'll have either the expected
            # state value or 0 in case the state was final.
            next_state_values = torch.zeros(self.hparams.batch_size, device=self.device)
            with torch.no_grad():
                next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
            # Compute the expected Q values
            expected_state_action_values = (next_state_values * self.hparams.gamma) + reward_batch

            # Compute Huber loss
            criterion = nn.SmoothL1Loss()
            loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

            # Optimize the model
            optimizer.zero_grad()
            loss.backward()
            # In-place gradient clipping
            torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
            optimizer.step()

            # Soft update of the target network's weights
            # these were after optimize_model
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()

            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[
                    key
                ] * self.TAU + target_net_state_dict[key] * (1 - self.TAU)
            target_net.load_state_dict(target_net_state_dict)
