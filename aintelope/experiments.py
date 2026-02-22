# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Repository:
# https://github.com/aintelope-london/attention-schema-theory-experiment

import gc
from pathlib import Path
from omegaconf import DictConfig

from aintelope.agents import get_agent_class
from aintelope.analytics.recording import (
    EventLog,
    StateLog,
    get_checkpoint,
    checkpoint_path,
)
from aintelope.environments import get_env_class
from aintelope.training.dqn_training import Trainer
from aintelope.utils.performance import ResourceMonitor
from aintelope.utils.roi import compute_roi


def _extract_roi_inputs(infos):
    """Pull positions and directions from augmented infos dict."""
    positions = {aid: infos[aid]["position"] for aid in infos}
    directions = {aid: infos[aid]["direction"] for aid in infos}
    return positions, directions


def run_experiment(
    cfg: DictConfig,
    score_dimensions: list = [],
    i_trial: int = 0,
    reporter=None,
) -> None:
    is_sb3 = cfg.agent_params.agent_class.startswith("sb3_")
    mode = cfg.env_params.mode
    roi_mode = cfg.agent_params.roi_mode
    radius = cfg.env_params.render_agent_radius

    monitor = ResourceMonitor(
        context={
            "trial": i_trial,
            "episodes": cfg.run.episodes,
            "steps": cfg.run.steps,
        }
    )

    # Environment
    env = get_env_class(cfg.env_params.env)(cfg=cfg)
    observations, infos = env.reset()
    manifesto = env.manifesto

    # Event logging — score dimensions from env, not hardcoded
    score_dims = env.score_dimensions
    events_columns = list(cfg.run.event_columns) + score_dims
    events = EventLog(events_columns)
    events.experiment_name = cfg.experiment_name
    states = StateLog()

    # Common trainer for each agent's models
    trainer = None if is_sb3 else Trainer(cfg)

    # Agents
    agents = []
    dones = {}

    for i in range(env.max_num_agents):
        agent_id = f"agent_{i}"
        agent = get_agent_class(cfg.agent_params.agent_class)(
            agent_id=agent_id,
            trainer=trainer,
            env=env,
            cfg=cfg,
            **cfg.agent_params,
        )
        agents.append(agent)

        if is_sb3:
            agent.i_trial = i_trial
            agent.events = events
            agent.score_dimensions = score_dims

        observation = observations[agent_id]
        info = infos[agent_id]

        # Legacy: observation shape depends on combine_interoception_and_vision
        if not cfg.env_params.combine_interoception_and_vision:
            obs_shape = (observation[0].shape, observation[1].shape)
        else:
            obs_shape = observation.shape

        print(f"\nAgent {agent_id} observation shape: {obs_shape}")

        agent.reset(observation, info, type(env))
        checkpoint = get_checkpoint(cfg.run.outputs_dir, agent_id)

        # Legacy: model init shape depends on observation format
        if not cfg.env_params.combine_interoception_and_vision:
            agent.init_model(
                (observation[0].shape, observation[1].shape),
                env.action_space(agent_id),
                checkpoint=checkpoint,
            )
        else:
            agent.init_model(
                observation.shape,
                env.action_space(agent_id),
                checkpoint=checkpoint,
            )
        dones[agent_id] = False

    # Main loop
    monitor.sample("init")

    # SB3 training has its own loop (special permission — documented in DOCUMENTATION.md)
    if is_sb3 and not cfg.run.test_mode:
        monitor.sample("sb3_train_start")
        run_baseline_training(cfg, i_trial, env, agents)
        monitor.sample("sb3_train_end")
        gc.collect()
        monitor.report(Path(cfg.run.outputs_dir) / cfg.experiment_name)
        return {"events": events.to_dataframe(), "states": states.to_dataframe()}

    reporter.set_total("episode", cfg.run.episodes)
    for i_episode in range(cfg.run.episodes):
        reporter.update("episode", i_episode + 1)

        env_layout_seed = (
            int(i_episode / cfg.env_params.env_layout_seed_repeat_sequence_length)
            if cfg.env_params.env_layout_seed_repeat_sequence_length > 0
            else i_episode
        )

        if cfg.env_params.env_layout_seed_modulo > 0:
            env_layout_seed = env_layout_seed % cfg.env_params.env_layout_seed_modulo

        print(
            f"\ni_trial: {i_trial} episode: {i_episode} "
            f"env_layout_seed: {env_layout_seed} test_mode: {cfg.run.test_mode}"
        )

        # Reset
        observations, infos = env.reset(env_layout_seed=env_layout_seed)

        for agent in agents:
            agent.reset(observations[agent.id], infos[agent.id], type(env))
            dones[agent.id] = False

        monitor.sample("reset")

        # Iterations within the episode
        for step in range(cfg.run.steps):
            # Collect actions from all agents
            actions = {}
            for agent in agents:
                actions[agent.id] = agent.get_action(
                    observation=observations[agent.id],
                    info=infos[agent.id],
                    step=step,
                    env_layout_seed=env_layout_seed,
                    episode=i_episode,
                    trial=i_trial,
                )

            # Step — one permitted branch on mode
            if mode == "parallel":
                (
                    observations,
                    scores,
                    terminateds,
                    truncateds,
                    infos,
                ) = env.step_parallel(actions)
            else:
                (
                    observations,
                    scores,
                    terminateds,
                    truncateds,
                    infos,
                ) = env.step_sequential(actions)

            dones = {aid: terminateds[aid] or truncateds[aid] for aid in terminateds}

            # ROI
            positions, directions = _extract_roi_inputs(infos)
            observations = compute_roi(
                observations, positions, directions, roi_mode, radius
            )

            # Update agents
            for agent in agents:
                observation = observations[agent.id]
                score = scores[agent.id]
                done = dones[agent.id]
                if terminateds[agent.id]:
                    observation = None
                agent_step_info = agent.update(
                    env=env,
                    observation=observation,
                    info=infos[agent.id],
                    score=sum(score.values()),
                    done=done,
                )

                # Record event
                env_step_info = [score.get(dim, 0) for dim in score_dims]
                # raw_obs = infos[agent.id]["raw_observation"]
                events.log_event(
                    [
                        cfg.experiment_name,
                        i_trial,
                        i_episode,
                        env_layout_seed,
                        step,
                        cfg.run.test_mode,
                    ]
                    + agent_step_info
                    # + [raw_obs]
                    + [observations[agent.id]]
                    + env_step_info
                )

            # Record global board state once per step
            board, layer_order = env.board_state()
            states.log(
                [cfg.experiment_name, i_trial, i_episode, step, (board, layer_order)]
            )

            # Perform one step of the optimization (on the policy network)
            if not cfg.run.test_mode:
                trainer.optimize_models()

            # Break when all agents are done
            if all(dones.values()):
                break

        monitor.sample("steps")

    if not cfg.run.test_mode:
        for agent in agents:
            agent.save_model(checkpoint_path(cfg.run.outputs_dir, agent.id))

    gc.collect()
    monitor.report(Path(cfg.run.outputs_dir) / cfg.experiment_name)
    return {"events": events.to_dataframe(), "states": states.to_dataframe()}


def run_baseline_training(cfg: DictConfig, i_trial: int, env, agents: list):
    num_total_steps = cfg.run.steps * cfg.run.episodes
    agents[0].train(num_total_steps)
    agents[0].save_model(checkpoint_path(cfg.run.outputs_dir, agents[0].id))
