# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Repository:
# https://github.com/aintelope-london/attention-schema-theory-experiment

import os
import traceback
from typing import Optional, Tuple
from gymnasium.spaces import Discrete
from pathlib import Path

import pandas as pd
from omegaconf import DictConfig, OmegaConf

import numpy as np
import numpy.typing as npt
import sys

from aintelope.config.config_utils import select_gpu, set_priorities, set_memory_limits

from aintelope.agents import AbstractAgent
from aintelope.aintelope_typing import ObservationFloat, PettingZooEnv

from aintelope.environments.savanna_safetygrid import (
    INFO_REWARD_DICT,
    INFO_AGENT_OBSERVATION_LAYERS_CUBE,
    INFO_AGENT_INTEROCEPTION_VECTOR,
)
from zoo_to_gym_multiagent_adapter.multiagent_zoo_to_gym_adapter import (
    MultiAgentZooToGymAdapterGymSide,
    MultiAgentZooToGymAdapterZooSide,
)

import stable_baselines3
from aintelope.agents.model.dl_utils import checkpoint_path

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from typing import Any, Union
import gymnasium as gym
from pettingzoo import AECEnv, ParallelEnv

# SB3-internal context keys injected into the policy's info dict.
# "episode" is reserved by Stable Baselines internally.
INFO_trial = "trial"
INFO_EPISODE = "i_episode"
INFO_ENV_LAYOUT_SEED = "env_layout_seed"
INFO_STEP = "step"
INFO_TEST_MODE = "test_mode"

PettingZooEnv = Union[AECEnv, ParallelEnv]
Environment = Union[gym.Env, PettingZooEnv]


class PolicyWithConfigFactory(object):
    def __init__(self, env_classname, agent_id, cfg, policy_constructor):
        self.env_classname = env_classname
        self.agent_id = agent_id
        self.cfg = cfg
        self.policy_constructor = policy_constructor

    def __call__(self, *args, **kwargs):
        return self.policy_constructor(
            self.env_classname, self.agent_id, self.cfg, *args, **kwargs
        )


def vec_env_args(env, num_envs):
    assert num_envs == 1

    def env_fn():
        env_copy = env  # TODO: assertion that this is called only once per env
        return env_copy

    return [env_fn] * num_envs, env.observation_space, env.action_space


def is_json_serializable(item: Any) -> bool:
    return False


# TODO: move to a separate file
class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256, num_conv_layers=2):
        super().__init__(observation_space, features_dim)

        num_channels = observation_space.shape[0]
        height = observation_space.shape[1]
        width = observation_space.shape[2]

        if num_conv_layers == 2:
            self.cnn = nn.Sequential(
                nn.Conv2d(num_channels, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Flatten(),
            )
        elif num_conv_layers == 3:
            self.cnn = nn.Sequential(
                nn.Conv2d(num_channels, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Flatten(),
            )
        else:
            self.cnn = nn.Sequential(
                nn.Flatten(),
            )

        with torch.no_grad():
            sample = torch.zeros(1, num_channels, height, width)
            n_flatten = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


def sb3_agent_train_thread_entry_point(env_wrapper, agent_id, model_constructor, cfg):
    try:
        select_gpu(cfg)
        set_priorities(cfg)
        set_memory_limits(cfg)

        env_gym_side = MultiAgentZooToGymAdapterGymSide(env_wrapper, agent_id)
        model = model_constructor(
            env_gym_side, env_wrapper.env_classname, agent_id, cfg
        )
        model.learn(total_timesteps=env_wrapper.num_total_steps)
        env_wrapper.set_model(agent_id, model)
    except Exception as ex:
        info = str(ex) + os.linesep + traceback.format_exc()
        env_wrapper.terminate_with_exception(info)
        print(info)


class SB3BaseAgent(AbstractAgent):
    """SB3BaseAgent — wraps Stable Baselines 3 models behind the AbstractAgent contract.

    Training runs through SB3's own loop (special permission, see DOCUMENTATION.md).
    Test mode runs through the canonical experiment loop.
    """

    def __init__(
        self,
        agent_id: str,
        env: Environment = None,
        cfg: DictConfig = None,
        **kwargs,
    ) -> None:
        self.id = agent_id
        self.cfg = cfg
        self.env = env
        self.env_classname = (
            env.__class__.__bases__[0].__module__
            + "."
            + env.__class__.__bases__[0].__qualname__
        )
        self.test_mode = self.cfg.run.experiment.test_mode
        self.i_trial = 0
        self.next_episode_no = 0
        self.total_steps_across_episodes = 0
        self.score_dimensions = []
        self.events = None
        self.state_log = None
        self.done = False
        self.last_action = None
        self.info = None
        self.state = None
        self.infos = {}
        self.states = {}
        self.model = None  # single-model scenario
        self.models = None  # multi-model scenario
        self.exceptions = None  # multi-model scenario
        self.model_constructor = None

        stable_baselines3.common.save_util.is_json_serializable = is_json_serializable

    def _prepare_obs(self, observation):
        """Convert multimodal dict observation to combined (C+N, H, W) ndarray.
        Each interoception value becomes a constant-filled spatial layer."""
        vision = observation["vision"]
        interoception = observation["interoception"]
        H, W = vision.shape[1], vision.shape[2]
        intero_layers = np.broadcast_to(
            interoception[:, None, None], (len(interoception), H, W)
        ).copy()
        return np.concatenate([vision, intero_layers], axis=0)

    def reset(self, state, **kwargs) -> None:
        """Reset agent state. Called at the start of each episode."""
        self.done = False
        self.last_action = None
        self.state = self._prepare_obs(state)
        self.info = None
        self.states = {self.id: self.state}
        self.infos = {self.id: self.info}

    def get_action(self, observation=None, **kwargs) -> Optional[int]:
        """Predict action. Runs during canonical test loop.

        Builds SB3-internal context dict from kwargs (step, episode, trial)
        without requiring the experiment loop to pass env-specific info.
        """
        if self.done:
            return None

        self.info = {
            INFO_trial: kwargs.get("trial", 0),
            INFO_EPISODE: kwargs.get("episode", 0),
            INFO_ENV_LAYOUT_SEED: kwargs.get("seed", 0),
            INFO_STEP: kwargs.get("step", 0),
            INFO_TEST_MODE: self.cfg.run.experiment.test_mode,
        }
        self.infos[self.id] = self.info

        if self.model and hasattr(self.model.policy, "set_info"):
            self.model.policy.set_info(self.info)

        observation = self._prepare_obs(observation)
        action, _states = self.model.predict(observation, deterministic=True)
        action = np.asarray(action).item()

        action_space = self.env.action_spaces[self.id]
        if isinstance(action_space, Discrete):
            min_action = action_space.start
        else:
            min_action = action_space.min_action
        assert action >= min_action

        self.state = observation
        self.states[self.id] = observation
        self.last_action = action
        return action

    def update(self, observation=None, **kwargs) -> list:
        """Takes observations and updates on perceived experiences."""
        assert self.last_action is not None

        next_state = self._prepare_obs(observation)
        score = kwargs.get("score", 0.0)
        done = kwargs.get("done", False)
        info = kwargs.get("info", {})

        event = [self.id, self.state, self.last_action, score, done, next_state]
        self.state = next_state
        self.info = info
        return event

    def train(self, num_total_steps):
        self.env._sb3_training = True
        self.env._scalarize_rewards = True
        self.env._pre_reset_callback2 = self.env_pre_reset_callback
        self.env._post_reset_callback2 = self.env_post_reset_callback
        self.env._pre_step_callback2 = self.env_pre_step_callback
        if isinstance(self.env, ParallelEnv):
            self.env._post_step_callback2 = self.parallel_env_post_step_callback
        else:
            self.env._post_step_callback2 = self.sequential_env_post_step_callback

        checkpoint_dir = Path(self.cfg.run.outputs_dir) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if self.model is not None:
            self.model.learn(total_timesteps=num_total_steps)
        else:
            checkpoint_filenames = {
                agent_id: str(checkpoint_dir / agent_id)
                for agent_id in self.env.possible_agents
            }

            OmegaConf.resolve(self.cfg)

            env_wrapper = MultiAgentZooToGymAdapterZooSide(
                self.env, self.cfg, self.env_classname
            )
            self.models, self.exceptions = env_wrapper.train(
                num_total_steps=num_total_steps,
                agent_thread_entry_point=sb3_agent_train_thread_entry_point,
                model_constructor=self.model_constructor,
                terminate_all_agents_when_one_excepts=True,
                checkpoint_filenames=checkpoint_filenames,
            )
        self.env._sb3_training = False
        self.env._scalarize_rewards = False
        self.env._pre_reset_callback2 = None
        self.env._post_reset_callback2 = None
        self.env._post_step_callback2 = None

        if self.exceptions:
            raise Exception(str(self.exceptions))

    def save_model(self, path, **kwargs):
        if self.model is not None:
            torch.save(self.model.get_parameters(), path)
        elif self.models:
            i_trial = kwargs.get("i_trial", 0)
            outputs_dir = str(Path(path).parent.parent)
            for agent_id, model in self.models.items():
                torch.save(
                    model.get_parameters(),
                    checkpoint_path(outputs_dir, agent_id, i_trial),
                )

    def init_model(
        self,
        observation_shape,
        action_space,
        checkpoint: Optional[str] = None,
    ):
        if checkpoint:
            use_cuda = torch.cuda.is_available() and torch.cuda.device_count() > 0
            device = torch.device("cuda" if use_cuda else "cpu")
            params = torch.load(checkpoint, map_location=device)
            self.model.set_parameters(params, device=device)

    # ── SB3 training callbacks ─────────────────────────────────────────────────
    # Second permitted location for layout seed computation (SB3 training path).

    def env_pre_reset_callback(self, seed, options, *args, **kwargs):
        assert seed is None

        i_episode = self.next_episode_no
        self.next_episode_no += 1

        repeat_len = self.cfg.env_params.env_layout_seed_repeat_sequence_length
        env_layout_seed = int(i_episode / repeat_len) if repeat_len > 0 else i_episode
        if self.cfg.env_params.env_layout_seed_modulo > 0:
            env_layout_seed = (
                env_layout_seed % self.cfg.env_params.env_layout_seed_modulo
            )

        kwargs["env_layout_seed"] = env_layout_seed
        return (True, seed, options, args, kwargs)

    def env_post_reset_callback(self, states, infos, seed, options, *args, **kwargs):
        self.state = states[self.id]
        self.info = infos[self.id]
        self.states = states
        self.infos = infos

        i_trial = self.i_trial
        i_episode = self.next_episode_no - 1
        env_layout_seed = self.env.get_env_layout_seed()

        for agent, info in infos.items():
            info[INFO_trial] = i_trial
            info[INFO_EPISODE] = i_episode
            info[INFO_ENV_LAYOUT_SEED] = env_layout_seed
            info[INFO_STEP] = 0
            info[INFO_TEST_MODE] = self.cfg.run.experiment.test_mode

        if self.model:
            if hasattr(self.model.policy, "my_reset"):
                self.model.policy.my_reset(self.state, self.info)
            if hasattr(self.model.policy, "set_info"):
                self.model.policy.set_info(self.info)

    def env_pre_step_callback(self, actions):
        return actions

    def parallel_env_post_step_callback(
        self,
        actions,
        next_states,
        scores,
        terminateds,
        truncateds,
        infos,
        *args,
        **kwargs,
    ):
        if self.events is None:
            return

        i_trial = self.i_trial
        i_episode = self.next_episode_no - 1
        env_layout_seed = self.env.get_env_layout_seed()
        step = self.env.get_step_no() - 1

        for agent, next_state in next_states.items():
            state = self.states[agent]
            action = actions.get(agent, None)
            action = np.asarray(action).item()
            info = infos[agent]
            score = scores[agent]
            score2 = info[INFO_REWARD_DICT]
            done = terminateds[agent] or truncateds[agent]

            info[INFO_trial] = i_trial
            info[INFO_EPISODE] = i_episode
            info[INFO_ENV_LAYOUT_SEED] = env_layout_seed
            info[INFO_STEP] = step
            info[INFO_TEST_MODE] = self.cfg.run.experiment.test_mode

            agent_step_info = [
                agent,
                state,
                action,
                score,
                done,
                next_state,
                (
                    info[INFO_AGENT_OBSERVATION_LAYERS_CUBE],
                    info[INFO_AGENT_INTEROCEPTION_VECTOR],
                ),
            ]

            env_step_info = (
                [score2.get(dimension, 0) for dimension in self.score_dimensions]
                if isinstance(score2, dict)
                else [score2]
            )

            self.events.log_event(
                [
                    self.cfg.experiment_name,
                    i_trial,
                    i_episode,
                    env_layout_seed,
                    step,
                    self.cfg.run.experiment.test_mode,
                ]
                + agent_step_info
                + [info.get("position"), info.get("food_position"), None]
                + env_step_info
            )

        self.states = next_states
        self.infos = infos

        if self.model and hasattr(self.model.policy, "set_info"):
            self.state = next_states[self.id]
            self.info = infos[self.id]
            self.model.policy.set_info(self.info)

        if self.state_log is not None:
            env_state = self.env.state
            self.state_log.log(
                [
                    self.cfg.experiment_name,
                    i_trial,
                    i_episode,
                    step,
                    (env_state["board"], env_state["layers"]),
                ]
            )

    def sequential_env_post_step_callback(
        self,
        agent,
        action,
        next_state,
        score,
        terminated,
        truncated,
        info,
        *args,
        **kwargs,
    ):
        if self.events is None:
            return

        self.total_steps_across_episodes += 1

        action = np.asarray(action).item()
        done = terminated or truncated
        score2 = info[INFO_REWARD_DICT]

        agent_step_info = [
            agent,
            self.state,
            action,
            score,
            done,
            next_state,
            (
                info[INFO_AGENT_OBSERVATION_LAYERS_CUBE],
                info[INFO_AGENT_INTEROCEPTION_VECTOR],
            ),
        ]

        self.state = next_state
        self.info = info

        env_step_info = (
            [score2.get(dimension, 0) for dimension in self.score_dimensions]
            if isinstance(score2, dict)
            else [score2]
        )

        i_trial = self.i_trial
        i_episode = self.next_episode_no - 1
        env_layout_seed = self.env.get_env_layout_seed()
        step = self.env.get_step_no() - 1

        self.info[INFO_trial] = i_trial
        self.info[INFO_EPISODE] = i_episode
        self.info[INFO_ENV_LAYOUT_SEED] = env_layout_seed
        self.info[INFO_STEP] = step
        self.info[INFO_TEST_MODE] = self.cfg.run.experiment.test_mode

        self.infos[self.id] = self.info

        if self.model and hasattr(self.model.policy, "set_info"):
            self.model.policy.set_info(self.info)

        self.events.log_event(
            [
                self.cfg.experiment_name,
                i_trial,
                i_episode,
                env_layout_seed,
                step,
                self.cfg.run.experiment.test_mode,
            ]
            + agent_step_info
            + [info.get("position"), info.get("food_position"), None]
            + env_step_info
        )

        if self.state_log is not None:
            env_state = self.env.state
            self.state_log.log(
                [
                    self.cfg.experiment_name,
                    i_trial,
                    i_episode,
                    step,
                    (env_state["board"], env_state["layers"]),
                ]
            )
