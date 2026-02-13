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
from aintelope.environments.savanna_safetygrid import GridworldZooBaseEnv
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

    # NB! gridsearch_trial_no is NOT saved to output data files. Instead, the individual trials are identified by the timestamp_pid_uuid available in the experiment folder name. This enables running gridsearch on multiple computers concurrently without having to worry about unique gridsearch trial numbers allocation and potential collisions.
    events_columns = list(cfg.hparams.run_params.event_columns) + (
        score_dimensions if isinstance(env, GridworldZooBaseEnv) else ["Score"]
    )

    experiment_dir = os.path.normpath(cfg.experiment_dir)
    events_fname = cfg.events_fname

    events = recording.EventLog(
        experiment_dir,
        events_fname,
        events_columns,
    )

    # Common trainer for each agent's models
    if is_sb3:
        trainer = None
    else:
        trainer = Trainer(cfg)

    # normalise slashes in paths. This is not mandatory, but will be cleaner to debug
    dir_out = os.path.normpath(cfg.log_dir)
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
            cfg.hparams.model_params.use_weight_sharing  # The reasoning for this condition is that even if the agents have similar roles, they still see each other on different observation layers. For example, the agent 0 is always on layer 0 and agent 1 is alwyas on layer 1, regardless of which agent is observing. Thus if they were trained on separate models then they need separate models also during test. When PPO weight sharing is enabled, then this is not an issue, because the shared model will learn to differentiate between the agents by looking at the agent currently visible in the center of the observation field. TODO: Swap the self-agent and other-agent layers in the environment side in such a manner that self-agent is always at same layer index for all agents and other-agent is also always at same layer index for all agents. Then we can enable this conditional branch for other models as well in scenarios where the agents have symmetric roles.
            and prev_agent_checkpoint is not None
            # and not use_separate_models_for_each_experiment   # if each experiment has separate models then the model of first agent will have same age as the model of second agent. In this case there is no reason to restrict the model of second agent to be equal of the first agent
        ):  # later experiments may have more agents    # TODO: configuration option for determining whether new agents can copy the checkpoints of earlier agents, and if so then specifically which agent's checkpoint to use
            checkpoint = prev_agent_checkpoint
        else:
            checkpoint_filename = agent_id
            if use_separate_models_for_each_experiment:
                checkpoint_filename += "-" + experiment_name
            checkpoints = glob.glob(
                os.path.join(dir_cp, checkpoint_filename + "-*")
            )  # NB! separate agent id from date explicitly in glob arguments using "-" since theoretically the agent id could be a two digit number and we do not want to match agent_10 while looking for agent_1
            if len(checkpoints) > 0:
                checkpoint = max(checkpoints, key=os.path.getctime)
                prev_agent_checkpoint = checkpoint
            elif (
                prev_agent_checkpoint is not None
            ):  # later experiments may have more agents    # TODO: configuration option for determining whether new agents can copy the checkpoints of earlier agents, and if so then specifically which agent's checkpoint to use
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

    # SB3 training has its own loop, hijack process here for this special case
    if is_sb3 and not cfg.hparams.test_mode:
        run_baseline_training(cfg, i_trial, env, agents)
        events.close()
        return events

    model_needs_saving = (
        False  # if no training episodes are specified then do not save models
    )
    #reporter.set_total("episode", cfg.hparams.episodes)
    for i_episode in range(cfg.hparams.episodes):
        #reporter.update("episode", i_episode + 1)
        events.flush()

        env_layout_seed = (
            int(i_episode / cfg.hparams.env_layout_seed_repeat_sequence_length)
            if cfg.hparams.env_layout_seed_repeat_sequence_length > 0
            else i_episode  # this ensures that during test episodes, env_layout_seed based map randomization seed is different from training seeds. The environment is re-constructed when testing starts. Without explicitly providing env_layout_seed, the map randomization seed would be automatically reset to env_layout_seed = 0, which would overlap with the training seeds.
        )

        # How many different layout seeds there should be overall? After given amount of seeds has been used, the seed will loop over to zero and repeat the seed sequence. Zero or negative modulo parameter value disables the modulo feature.
        if cfg.hparams.env_layout_seed_modulo > 0:
            env_layout_seed = env_layout_seed % cfg.hparams.env_layout_seed_modulo

        print(
            f"\ni_trial: {i_trial} experiment: {experiment_name} episode: {i_episode} env_layout_seed: {env_layout_seed}"
        )

        # TODO: refactor these checks into separate function        # Save models
        # https://pytorch.org/tutorials/recipes/recipes/
        # saving_and_loading_a_general_checkpoint.html
        if not cfg.hparams.test_mode:
            if (
                i_episode > 0 and cfg.hparams.save_frequency != 0
            ):  # cfg.hparams.save_frequency == 0 means that the model is saved only at the end, improving training performance
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
            ) = env.reset(
                env_layout_seed=env_layout_seed
            )  # if not test_mode else -(env_layout_seed - cfg.hparams.episodes + 1))
            for agent in agents:
                agent.reset(observations[agent.id], infos[agent.id], type(env))
                # trainer.reset_agent(agent.id)	# TODO: configuration flag
                dones[agent.id] = False

        elif isinstance(env, AECEnv):
            env.reset(  # TODO: actually savanna_safetygrid wrapper provides observations and infos as a return value, so need for branching here
                env_layout_seed=env_layout_seed
            )  # if not test_mode else -(env_layout_seed - cfg.hparams.episodes + 1))
            for agent in agents:
                agent.reset(
                    env.observe(agent.id), env.observe_info(agent.id), type(env)
                )
                # trainer.reset_agent(agent.id)	# TODO: configuration flag
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
                # call update since the list of terminateds will become smaller on
                # second step after agents have died
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
                        score=sum(score.values())
                        if isinstance(score, dict)
                        else score,  # TODO: make a function to handle obs->rew in Q-agent too, remove this
                        done=done,  # TODO: should it be "terminated" in place of "done" here?
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
                # loop: observe, collect action, send action, get observation, update
                agents_dict = {agent.id: agent for agent in agents}
                for agent_id in env.agent_iter(
                    max_iter=env.num_agents
                ):  # num_agents returns number of alive (non-done) agents
                    agent = agents_dict[agent_id]

                    # Per Zoo API, a dead agent must call .step(None) once more after
                    # becoming dead. Only after that call will this dead agent be
                    # removed from various dictionaries and from .agent_iter loop.
                    if env.terminations[agent.id] or env.truncations[agent.id]:
                        action = None
                    else:
                        observation = env.observe(agent.id)
                        info = env.observe_info(agent.id)
                        action = agent.get_action(
                            observation=observation,
                            info=info,
                            step=step,
                            env_layout_seed=env_layout_seed,
                            episode=i_episode,
                            trial=i_trial,
                        )

                    # Env step
                    # NB! both AIntelope Zoo and Gridworlds Zoo wrapper in AIntelope
                    # provide slightly modified Zoo API. Normal Zoo sequential API
                    # step() method does not return values and is not allowed to return
                    # values else Zoo API tests will fail.
                    result = env.step_single_agent(action)

                    if agent.id in env.agents:  # was not "dead step"
                        # NB! This is only initial reward upon agent's own step.
                        # When other agents take their turns then the reward of the
                        # agent may change. If you need to learn an agent's accumulated
                        # reward over other agents turns (plus its own step's reward)
                        # then use env.last property.
                        (
                            observation,
                            score,
                            terminated,
                            truncated,
                            info,
                        ) = result

                        done = terminated or truncated

                        # Agent is updated based on what the env shows.
                        # All commented above included ^
                        if terminated:
                            observation = None  # TODO: why is this here?

                        agent_step_info = agent.update(
                            env=env,
                            observation=observation,
                            info=info,
                            score=sum(score.values())
                            if isinstance(score, dict)
                            else score,
                            done=done,  # TODO: should it be "terminated" in place of "done" here?
                        )  # note that score is used ONLY by baseline

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
                                env.terminations[agent_id2]
                                or env.truncations[agent_id2]
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

    events.close()

    return events


def run_baseline_training(
    cfg: DictConfig, i_trial: int, env: Environment, agents: list
):
    # SB3 models are designed for single-agent settings, we get around this by using the same model for every agent
    # https://pettingzoo.farama.org/tutorials/sb3/waterworld/

    # num_total_steps = cfg.hparams.env_params.num_iters * 1
    num_total_steps = cfg.hparams.env_params.num_iters * cfg.hparams.episodes

    # During multi-agent multi-model training the actual agents will run in threads/subprocesses because SB3 requires Gym interface. Agent[0] will be used just as an interface to call train(), the SB3BaseAgent base class will automatically set up the actual agents.
    # In case of multi-agent weight-shared model training it is partially similar: Agent[0] will be used just as an interface to call train(), the SB3 weight-shared model will handle the actual agents present in the environment.

    agents[0].train(num_total_steps)

    # Save models
    # for agent in agents:
    #    agent.save_model()
    agents[0].save_model()


if __name__ == "__main__":
    run_experiment()  # TODO: cfg, score_dimensions
