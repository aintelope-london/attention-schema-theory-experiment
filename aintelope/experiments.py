from collections import namedtuple

import logging
from omegaconf import DictConfig
import hydra
#import torch
#import torch.nn as nn
#import torch.optim as optim

from aintelope.environments.savanna_gym import SavannaGymEnv
from aintelope.models.dqn import DQN
from aintelope.agents import (
    Agent,
    GymEnv,
    PettingZooEnv,
    Environment,
    register_agent_class,
)
from aintelope.agents.instinct_agent import QAgent  # initialize agent registry
from aintelope.agents import get_agent_class
from aintelope.training.dqn_training import Trainer # TODO trainer as parameter
# from aintelope.agents.memory import ReplayBuffer, Experience

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@hydra.main(version_base=None, config_path="config", config_name="config_experiment")
def main(cfg: DictConfig) -> None:
    logger = logging.getLogger("aintelope.experiment")
    
    episode_durations = []

    # Environment
    env = SavannaGymEnv(env_params=cfg.hparams.env_params) #TODO: get env from parameters
    action_space = env.action_space # TODO what is this used for?
    observation, info = env.reset() #TODO: each agent has their own state, refactor
    # TODO: env doesnt register agents properly, it hallucinates from zooapi and names in its own way
    # figure out how to make this coherent. there's "possible_agents" now
    
    n_observations = len(observation) #len(state)  # get observation space from env
    #action_space = env.action_space # removed dynamic actions per agent for now
    
    # Common trainer for each agent's models
    trainer = Trainer(cfg, n_observations, action_space) # TODO: have a section in params for trainer? its trainer and hparams now tho
    
    # Agents
    #agents = {}
    #for i in range(cfg.params.amount_agents?):
    i = 0
    agent_id = f"agent_{i}"
    #    agents["agent_"+agent_id] = 
    agent = get_agent_class(cfg.hparams.agent_id)(
        agent_id,
        trainer,
        cfg.hparams.warm_start_steps,
        **cfg.hparams.agent_params,
    )
    # TODO: savanna_gym interface will reduce {agent_0:obs} to obs... take into account here
    agent.reset(env.observe(agent.id))
    trainer.add_agent(agent.id)
    #.... end of for-loop
    
    # buffer should be associated with agent
    #replay_buffer = ReplayBuffer(cfg.hparams.replay_size)
    # generalize to multi agent setup

    # networks should be part of the agent
    #policy_net = DQN(n_observations, n_actions).to(device)
    #target_net = DQN(n_observations, n_actions).to(device)
    #optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    #target_net.load_state_dict(policy_net.state_dict())
    
    # Warmup not supported atm, maybe not needed?
    #for _ in range(hparams.warm_start_steps):
    #     agents.play_step(self.net, epsilon=1.0) # TODO
    
    steps_done = 0
    num_episodes = 1 # TODO add episodes and modify epochs away in config. also, saving models! 
    for i_episode in range(num_episodes):
        # Reset
        state, info = env.reset() # remove state and info from here
        #for agent in agents:
        agent.reset(env.observe(agent.id))
        #state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        for step in range(1): #TODO
            '''
            # set epsilon value. should be in agent
            epsilon = max(
                cfg.hparams.eps_end,
                cfg.hparams.eps_start - step * 1 / cfg.hparams.eps_last_frame,
            )
            '''
            # for agent in agents:
            observation = env.observe(agent.id)
            action = agent.get_action(observation, step)#state, policy_net, epsilon, "cpu")
            
            # Env step
            if isinstance(env, GymEnv):
                observation, score, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
            elif isinstance(env, PettingZooEnv):
                observation, score, terminateds, truncateds, _ = env.step(action)
                done = {
                    key: terminated or truncateds[key]
                    for (key, terminated) in terminateds.items()
                }
            else:
                logger.warning(f"{env} is not of type GymEnv or PettingZooEnv")
                observation, score, done, _ = env.step(action)
            ### TODO: move to support only pettingzoo?
            #observation, reward, terminated, truncated, _ = env.step(action)
            
            '''
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated
            
            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(
                    observation, dtype=torch.float32, device=device
                ).unsqueeze(0)

            # Store the transition in memory
            exp = Experience(state, action, reward, done, next_state)
            replay_buffer.append(exp)

            # Move to the next state
            state = next_state
            '''
            # Agent is updated based on what the env shows. All commented above included ^
            done = terminated or truncated
            agent.update(env, observation, score, done) # note that score is used ONLY by baseline
            
            # Perform one step of the optimization (on the policy network)
            # TEST: if we call this every time, will it overlearn the initial steps? The buffer
            # is filled only with a batch worth of stuff, and it might overrepresent?
            trainer.optimize_models(step)

            if done:
                episode_durations.append(step + 1)
                break
'''
# WIP: instantiate?
def reset() -> None:
    """Resets environment and agents."""
    self.done = False
    self.state = state
    if isinstance(self.state, tuple):
        self.state = self.state[0]
'''   
        
if __name__ == "__main__":
    main()
