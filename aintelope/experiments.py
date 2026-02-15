# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Repository:
# https://github.com/biological-alignment-benchmarks/biological-alignment-gridworlds-benchmarks

import glob

import os
import gc

from omegaconf import DictConfig

from aintelope.agents import get_agent_class
from aintelope.analytics import recording
from aintelope.environments import get_env_class
from aintelope.environments.savanna_safetygrid import (
    GridworldZooBaseEnv,
    INFO_AGENT_OBSERVATION_LAYERS_ORDER,
)
from aintelope.training.dqn_training import Trainer

from typing import Union
import gymnasium as gym
from pettingzoo import AECEnv, ParallelEnv

PettingZooEnv = Union[AECEnv, ParallelEnv]
Environment = Union[gym.Env, PettingZooEnv]


def run_experiment(
    cfg: DictConfig,
    experiment_name: str = "",  # TODO: remove this argument and read it from cfg.experiment_name
    score_dimensions: list = [],
    i_trial: int = 0,
    reporter=None,
) -> None:
    if "trial_length" in cfg:  # backwards compatibility
        cfg.env_layout_seed_repeat_sequence_length = cfg.trial_length
    if "eps_last_env_layout_seed" in cfg:  # backwards compatibility
        cfg.eps_last_env_layout_seed = cfg.eps_last_env_layout_seed

    is_sb3 = cfg.hparams.agent_class.startswith("sb3_")

    # Environment
    env = get_env_class(cfg.hparams.env)(cfg=cfg)

    # This reset here does not increment episode number since no steps are played before one more reset in the main episode loop takes place
    if isinstance(env, ParallelEnv):
        (
            observations,
            infos,
        ) = env.reset()
    elif isinstance(env, AECEnv):
        env.reset()
    else:
        raise NotImplementedError(f"Unknown environment type {type(env)}")

    events_columns = list(cfg.hparams.run_params.event_columns) + (
        score_dimensions if isinstance(env, GridworldZooBaseEnv) else ["Score"]
    )

    events = recording.EventLog(events_columns)

    # Capture observation layer order for playback rendering
    first_agent_info = (
        infos["agent_0"]
        if isinstance(env, ParallelEnv)
        else env.observe_info("agent_0")
    )
    events.metadata["layer_order"] = first_agent_info[
        INFO_AGENT_OBSERVATION_LAYERS_ORDER
    ]

    # Common trainer for each agent's models
    if is_sb3:
        trainer = None
    else:
        trainer = Trainer(cfg)

    dir_out = os.path.normpath(cfg.outputs_dir)
    checkpoint_dir = os.path.normpath(cfg.checkpoint_dir)
    dir_cp = os.path.join(dir_out, checkpoint_dir)

    use_separate_models_for_each_experiment = (
        cfg.hparams.use_separate_models_for_each_experiment
    )

    # Agents
    agents = []
    dones = {}
    prev_agent_checkpoint = None
    for i in range(env.max_num_agents):
        agent_id = f"agent_{i}"
        agent = get_agent_class(cfg.hparams.agent_class)(
            agent_id=agent_id,
            trainer=trainer,
            env=env,
            cfg=cfg,
            **cfg.hparams.agent_params,
        )
        agents.append(agent)

        if is_sb3:
            agent.i_trial = i_trial
            agent.events = events
            agent.score_dimensions = score_dimensions

        # TODO: IF agent.reset() below is not needed then it is possible to call
        # env.observation_space(agent_id) directly to get the observation shape.
        # No need to call observe().
        if isinstance(env, ParallelEnv):
            observation = observations[agent_id]
            info = infos[agent_id]
        elif isinstance(env, AECEnv):
            observation = env.observe(agent_id)
            info = env.observe_info(agent_id)

        if not cfg.hparams.env_params.combine_interoception_and_vision:
            observation_shape = (observation[0].shape, observation[1].shape)
        else:
            observation_shape = observation.shape

        print(f"\nAgent {agent_id} observation shape: {observation_shape}")

        # TODO: is this reset necessary here? In main loop below,
        # there is also a reset call
        agent.reset(observation, info, type(env))
        # Get latest checkpoint if existing

        checkpoint = None

        if (
            cfg.hparams.model_params.use_weight_sharing
            and prev_agent_checkpoint is not None
        ):
            checkpoint = prev_agent_checkpoint
        else:
            checkpoint_filename = agent_id
            if use_separate_models_for_each_experiment:
                checkpoint_filename += "-" + experiment_name
            checkpoints = glob.glob(os.path.join(dir_cp, checkpoint_filename + "-*"))
            if len(checkpoints) > 0:
                checkpoint = max(checkpoints, key=os.path.getctime)
                prev_agent_checkpoint = checkpoint
            elif prev_agent_checkpoint is not None:
                checkpoint = prev_agent_checkpoint

        # Add agent, with potential checkpoint
        if not cfg.hparams.env_params.combine_interoception_and_vision:
            agent.init_model(
                (observation[0].shape, observation[1].shape),
                env.action_space,
                checkpoint=checkpoint,
            )
        else:
            agent.init_model(
                observation.shape,
                env.action_space,
                checkpoint=checkpoint,
            )
        dones[agent_id] = False

    # Main loop

    # SB3 training has its own loop
    if is_sb3 and not cfg.hparams.test_mode:
        run_baseline_training(cfg, i_trial, env, agents)
        gc.collect()
        return events

    model_needs_saving = (
        False  # if no training episodes are specified then do not save models
    )
    reporter.set_total("episode", cfg.hparams.episodes)
    for i_episode in range(cfg.hparams.episodes):
        reporter.update("episode", i_episode + 1)

        env_layout_seed = (
            int(i_episode / cfg.hparams.env_layout_seed_repeat_sequence_length)
            if cfg.hparams.env_layout_seed_repeat_sequence_length > 0
            else i_episode
        )

        if cfg.hparams.env_layout_seed_modulo > 0:
            env_layout_seed = env_layout_seed % cfg.hparams.env_layout_seed_modulo

        print(
            f"\ni_trial: {i_trial} experiment: {experiment_name} episode: {i_episode} env_layout_seed: {env_layout_seed} test_mode: {cfg.hparams.test_mode}"
        )

        # TODO: refactor these checks into separate function
        # Save models
        if not cfg.hparams.test_mode:
            if i_episode > 0 and cfg.hparams.save_frequency != 0:
                model_needs_saving = True
                if i_episode % cfg.hparams.save_frequency == 0:
                    os.makedirs(dir_cp, exist_ok=True)
                    for agent in agents:
                        agent.save_model(
                            i_episode,
                            dir_cp,
                            experiment_name,
                            use_separate_models_for_each_experiment,
                        )

                    model_needs_saving = False
            else:
                model_needs_saving = True

        # Reset
        if isinstance(env, ParallelEnv):
            (
                observations,
                infos,
            ) = env.reset(env_layout_seed=env_layout_seed)
            for agent in agents:
                agent.reset(observations[agent.id], infos[agent.id], type(env))
                dones[agent.id] = False

        elif isinstance(env, AECEnv):
            env.reset(env_layout_seed=env_layout_seed)
            for agent in agents:
                agent.reset(
                    env.observe(agent.id), env.observe_info(agent.id), type(env)
                )
                dones[agent.id] = False

        # Iterations within the episode
        for step in range(cfg.hparams.env_params.num_iters):
            if isinstance(env, ParallelEnv):
                # loop: get observations and collect actions
                actions = {}
                for agent in agents:  # TODO: exclude terminated agents
                    observation = observations[agent.id]
                    info = infos[agent.id]
                    actions[agent.id] = agent.get_action(
                        observation=observation,
                        info=info,
                        step=step,
                        env_layout_seed=env_layout_seed,
                        episode=i_episode,
                        trial=i_trial,
                    )

                # call: send actions and get observations
                (
                    observations,
                    scores,
                    terminateds,
                    truncateds,
                    infos,
                ) = env.step(actions)
                dones.update(
                    {
                        key: terminated or truncateds[key]
                        for (key, terminated) in terminateds.items()
                    }
                )

                # loop: update
                for agent in agents:
                    observation = observations[agent.id]
                    info = infos[agent.id]
                    score = scores[agent.id]
                    done = dones[agent.id]
                    terminated = terminateds[agent.id]
                    if terminated:
                        observation = None
                    agent_step_info = agent.update(
                        env=env,
                        observation=observation,
                        info=info,
                        score=sum(score.values()) if isinstance(score, dict) else score,
                        done=done,
                    )

                    # Record what just happened
                    env_step_info = (
                        [score.get(dimension, 0) for dimension in score_dimensions]
                        if isinstance(score, dict)
                        else [score]
                    )

                    events.log_event(
                        [
                            cfg.experiment_name,
                            i_trial,
                            i_episode,
                            env_layout_seed,
                            step,
                            cfg.hparams.test_mode,
                        ]
                        + agent_step_info
                        + env_step_info
                    )

            elif isinstance(env, AECEnv):
                agents_dict = {agent.id: agent for agent in agents}
                for agent_id in env.agent_iter(max_iter=env.num_agents):
                    agent = agents_dict[agent_id]

                    if dones[agent_id]:
                        env.step(None)
                        continue

                    observation = env.observe(agent_id)
                    info = env.observe_info(agent_id)
                    action = agent.get_action(
                        observation=observation,
                        info=info,
                        step=step,
                        env_layout_seed=env_layout_seed,
                        episode=i_episode,
                        trial=i_trial,
                    )

                    env.step(action)

                    observation = env.observe(agent_id)
                    info = env.observe_info(agent_id)
                    score = env.rewards[agent_id]
                    done = env.terminations[agent_id] or env.truncations[agent_id]

                    if env.terminations[agent_id]:
                        observation = None

                    agent_step_info = agent.update(
                        env=env,
                        observation=observation,
                        info=info,
                        score=sum(score.values()) if isinstance(score, dict) else score,
                        done=done,
                    )

                    # Record what just happened
                    env_step_info = (
                        [score.get(dimension, 0) for dimension in score_dimensions]
                        if isinstance(score, dict)
                        else [score]
                    )

                    events.log_event(
                        [
                            cfg.experiment_name,
                            i_trial,
                            i_episode,
                            env_layout_seed,
                            step,
                            cfg.hparams.test_mode,
                        ]
                        + agent_step_info
                        + env_step_info
                    )

                    # NB! any agent could die at any other agent's step
                    for agent_id2 in env.agents:
                        dones[agent_id2] = (
                            env.terminations[agent_id2] or env.truncations[agent_id2]
                        )
                        # TODO: if the agent died during some other agents step,
                        # should we call agent.update() on the dead agent,
                        # else it will be never called?

            else:
                raise NotImplementedError(f"Unknown environment type {type(env)}")

            # Perform one step of the optimization (on the policy network)
            if not cfg.hparams.test_mode:
                trainer.optimize_models()

            # Break when all agents are done
            if all(dones.values()):
                break

        if (
            model_needs_saving
        ):  # happens when episodes is not divisible by save frequency
            os.makedirs(dir_cp, exist_ok=True)
            for agent in agents:
                agent.save_model(
                    i_episode,
                    dir_cp,
                    experiment_name,
                    use_separate_models_for_each_experiment,
                )

    gc.collect()

    return events


def run_baseline_training(
    cfg: DictConfig, i_trial: int, env: Environment, agents: list
):
    num_total_steps = cfg.hparams.env_params.num_iters * cfg.hparams.episodes

    agents[0].train(num_total_steps)

    agents[0].save_model()


if __name__ == "__main__":
    run_experiment()  # TODO: cfg, score_dimensions
