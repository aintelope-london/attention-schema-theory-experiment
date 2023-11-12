from collections import namedtuple

from omegaconf import DictConfig
import hydra
import torch
import torch.nn as nn
import torch.optim as optim

from aintelope.environments.savanna_gym import SavannaGymEnv
from aintelope.models.dqn import DQN
from aintelope.agents.instinct_agent import QAgent  # initialize agent registry
from aintelope.agents import get_agent_class
from aintelope.agents.memory import ReplayBuffer, Experience


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# replace by experience
Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4


def optimize_model(optimizer, memory, policy_net, target_net):
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=device,
        dtype=torch.bool,
    )
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


if torch.cuda.is_available():
    num_episodes = 600
else:
    num_episodes = 1


@hydra.main(version_base=None, config_path="config", config_name="config_experiment")
def main(cfg: DictConfig) -> None:
    episode_durations = []

    env = SavannaGymEnv(env_params=cfg.hparams.env_params)
    n_actions = env.action_space.n
    state, info = env.reset()
    n_observations = len(state)  # get observation space from env

    # buffer should be associated with agent
    replay_buffer = ReplayBuffer(cfg.hparams.replay_size)
    # generalize to multi agent setup
    agent = get_agent_class(agent_id=cfg.hparams.agent_id)(env, replay_buffer, 0)

    # networks should be part of the agent
    policy_net = DQN(n_observations, n_actions).to(device)
    target_net = DQN(n_observations, n_actions).to(device)
    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    target_net.load_state_dict(policy_net.state_dict())

    steps_done = 0

    for i_episode in range(num_episodes):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        for t in range(1):
            # set epsilon value
            epsilon = max(
                cfg.hparams.eps_end,
                cfg.hparams.eps_start - t * 1 / cfg.hparams.eps_last_frame,
            )

            # should be for multiple agents, e.g. via a loop
            action = agent.get_action(state, policy_net, epsilon, "cpu")
            observation, reward, terminated, truncated, _ = env.step(action)
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

            # Perform one step of the optimization (on the policy network)
            optimize_model(optimizer, replay_buffer, policy_net, target_net)

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()

            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[
                    key
                ] * TAU + target_net_state_dict[key] * (1 - TAU)
            target_net.load_state_dict(target_net_state_dict)

            if done:
                episode_durations.append(t + 1)
                break


if __name__ == "__main__":
    main()
