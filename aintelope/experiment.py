# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import gc
import numpy as np
from pathlib import Path
from omegaconf import DictConfig

from aintelope.agents import get_agent_class
from aintelope.agents.model.dl_utils import checkpoint_path, select_checkpoint
from aintelope.analytics.diagnostics import DiagnosticsMonitor
from aintelope.analytics.recording import EventLog, StateLog, save_env_layout
from aintelope.environments import get_env_class


def run_experiment(
    cfg: DictConfig,
    i_trial: int = 0,
    reporter=None,
) -> dict:
    is_sb3 = cfg.agent_params.agents.agent_0.agent_class.startswith("sb3_")

    monitor = DiagnosticsMonitor(
        context={
            "trial": i_trial,
            "episodes": cfg.run.experiment.episodes,
            "steps": cfg.run.experiment.steps,
        }
    )

    # Environment
    env = get_env_class(cfg.env_params.env)(cfg=cfg)
    observations, state = env.reset()

    # Event logging
    score_dims = env.score_dimensions
    events = EventLog(list(cfg.run.experiment.event_columns) + score_dims)
    events.experiment_name = cfg.experiment_name
    states = StateLog()

    # Agents
    agents = []
    dones = {}
    custom_model = cfg.agent_params.get("custom_model", "")

    if is_sb3:
        agents, dones = _init_sb3_agents(
            cfg, env, observations, state, events, score_dims, i_trial, custom_model
        )
    else:
        for agent_id, agent_cfg in cfg.agent_params.agents.items():
            checkpoint = select_checkpoint(
                cfg.run.outputs_dir, agent_id, i_trial, custom_model
            )
            agent = get_agent_class(agent_cfg.agent_class)(
                agent_id=agent_id,
                env=env,
                cfg=cfg,
                checkpoint=checkpoint,
            )
            agents.append(agent)
            agent.reset(observations[agent_id])
            dones[agent_id] = False

    monitor.sample("init")

    # SB3 training has its own loop (special permission — documented in DOCUMENTATION.md)
    if is_sb3 and not cfg.run.experiment.test_mode:
        _run_sb3_training(cfg, i_trial, env, agents, events, states, monitor)
        return {
            "events": events.to_dataframe(),
            "states": states.to_dataframe(),
            "learning_df": monitor.learning_dataframe(),
            "performance_df": monitor.performance_dataframe(),
            "manifesto": env.manifesto,
        }

    save_freq = cfg.agent_params.save_frequency
    states_by_seed = {}  # seed → state at reset; used for layout rendering

    reporter.set_total("episode", cfg.run.experiment.episodes)
    for i_episode in range(cfg.run.experiment.episodes):
        reporter.update("episode", i_episode + 1)

        observations, state = env.reset(seed=i_episode)
        states_by_seed[i_episode] = state
        # Retain food position from episode start — remains valid even after food consumed.
        # Analytics use this coordinate to track when the agent reaches it.
        episode_food_position = state.get("food_position")

        # Log reset state at step -1 — canonical episode start before any action.
        states.log(
            [
                cfg.experiment_name,
                i_trial,
                i_episode,
                -1,
                (state["board"], state["layers"], None),
            ]
        )

        for agent in agents:
            agent.reset(observations[agent.id])
            dones[agent.id] = False

        monitor.sample("reset")

        for step in range(cfg.run.experiment.steps):
            actions = {}
            for agent in agents:
                result = agent.get_action(
                    observation=observations[agent.id],
                    step=step,
                    episode=i_episode,
                    trial=i_trial,
                )
                actions[agent.id] = (
                    result if isinstance(result, dict) else {"action": result}
                )

            observations, state = env.step_parallel(actions)
            dones = state["dones"]

            for agent in agents:
                observation = observations[agent.id]
                done = dones[agent.id]
                report = agent.update(observation=observation, done=done)
                monitor.sample_learning(i_episode, step, report)

                events.log_event(
                    [
                        cfg.experiment_name,
                        i_trial,
                        i_episode,
                        i_episode,
                        step,
                        cfg.run.experiment.test_mode,
                        agent.id,
                        agent.last_action,
                        report.get("reward", 0),  # RI reward — canonical signal
                        done,
                        observation,
                        state["agent_positions"].get(agent.id),  # post-step position
                        episode_food_position,  # retained from episode start
                        actions[agent.id].get("internal_action"),
                    ]
                )

            states.log(
                [
                    cfg.experiment_name,
                    i_trial,
                    i_episode,
                    step,
                    (state["board"], state["layers"], None),
                ]
            )

            if all(dones.values()):
                break

        monitor.sample("steps")

        if cfg.run.write_outputs and save_freq > 0 and (i_episode + 1) % save_freq == 0:
            for agent in agents:
                agent.save_model(
                    checkpoint_path(cfg.run.outputs_dir, agent.id, i_trial)
                )

    gc.collect()
    monitor.report()
    if cfg.run.write_outputs:
        for agent in agents:
            agent.save_model(checkpoint_path(cfg.run.outputs_dir, agent.id, i_trial))
        from aintelope.gui.renderer import (
            StateRenderer,
            Tileset,
            find_tileset,
            SavannaInterpreter,
        )

        renderer = StateRenderer(Tileset(find_tileset()))
        interpreter = SavannaInterpreter()
        for seed, s in states_by_seed.items():
            img = renderer.render(*interpreter.interpret((s["board"], s["layers"])))
            save_env_layout(img, Path(cfg.run.outputs_dir) / cfg.experiment_name, seed)

    return {
        "events": events.to_dataframe(),
        "states": states.to_dataframe(),
        "learning_df": monitor.learning_dataframe(),
        "performance_df": monitor.performance_dataframe(),
        "manifesto": env.manifesto,
    }


# ── SB3 legacy ────────────────────────────────────────────────────────


def _init_sb3_agents(
    cfg, env, observations, state, events, score_dims, i_trial, custom_model
):
    """Initialise SB3 agents with legacy interface. Isolated for readability."""
    agents = []
    dones = {}

    for agent_id, agent_cfg in cfg.agent_params.agents.items():
        agent = get_agent_class(agent_cfg.agent_class)(
            agent_id=agent_id,
            env=env,
            cfg=cfg,
        )
        agents.append(agent)
        agent.i_trial = i_trial
        agent.events = events
        agent.score_dimensions = score_dims

        agent.reset(observations[agent_id])
        checkpoint = select_checkpoint(
            cfg.run.outputs_dir, agent_id, i_trial, custom_model
        )
        agent.init_model(
            agent.state.shape,
            env.action_space(agent_id),
            checkpoint=checkpoint,
        )
        dones[agent_id] = False

    return agents, dones


def _run_sb3_training(cfg, i_trial, env, agents, events, states, monitor):
    """SB3 training loop — special permission documented in DOCUMENTATION.md."""
    monitor.sample("sb3_train_start")
    num_total_steps = cfg.run.experiment.steps * cfg.run.experiment.episodes
    agents[0].state_log = states
    agents[0].train(num_total_steps)
    monitor.sample("sb3_train_end")
    gc.collect()
    monitor.report()
    if cfg.run.write_outputs:
        agents[0].save_model(
            checkpoint_path(cfg.run.outputs_dir, agents[0].id, i_trial),
            i_trial=i_trial,
        )
