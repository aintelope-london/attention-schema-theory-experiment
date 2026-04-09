# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import gc
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
    monitor = DiagnosticsMonitor(
        context={
            "trial": i_trial,
            "episodes": cfg.run.experiment.episodes,
            "steps": cfg.run.experiment.steps,
        }
    )

    env = get_env_class(cfg.env_params.env)(cfg=cfg)
    observations, state = env.reset()

    score_dims = env.score_dimensions
    events = EventLog(list(cfg.run.experiment.event_columns))
    events.experiment_name = cfg.experiment_name
    states = StateLog()

    agents = []
    dones = {}
    custom_model = cfg.agent_params.get("custom_model", "")

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

    # SB3 training has its own loop (special permission — documented in DOCUMENTATION.md).
    if cfg.agent_params.agents[agents[0].id].agent_class == "sb3_agent":
        from aintelope.agents.sb3_agent import SB3Agent

        SB3Agent.training(
            env, cfg.run.experiment.steps * cfg.run.experiment.episodes, cfg, i_trial
        )
        gc.collect()
        monitor.report()
        return {
            "events": events.to_dataframe(),
            "states": states.to_dataframe(),
            "learning_df": monitor.learning_dataframe(),
            "performance_df": monitor.performance_dataframe(),
            "manifesto": env.manifesto,
        }

    save_freq = cfg.agent_params.save_frequency
    states_by_seed = {}

    reporter.set_total("episode", cfg.run.experiment.episodes)
    for i_episode in range(cfg.run.experiment.episodes):
        reporter.update("episode", i_episode + 1)

        observations, state = env.reset(seed=i_episode)
        states_by_seed[i_episode] = state
        episode_food_position = state.get("food_position")

        states.log([cfg.experiment_name, i_trial, i_episode, -1, state])

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
                        report.get("reward", 0),
                        done,
                        observation,
                        state["agent_positions"].get(agent.id),
                        episode_food_position,
                        actions[agent.id].get("internal_action"),
                    ]
                )

            states.log([cfg.experiment_name, i_trial, i_episode, step, state])

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
            Interpreter,
        )

        renderer = StateRenderer(Tileset(find_tileset()))
        interpreter = Interpreter(env.render_manifest)
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
