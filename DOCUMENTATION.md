# DOCUMENTATION

## Overview

This project provides an experimental orchestrator for running arbitrary agents in arbitrary environments that conform to the MOMA MDP (Multi-Objective Multi-Agent Markov Decision Process) structure. Agents and environments are registered via a factory pattern, making the system agnostic to their implementation. The orchestrator handles configuration, trial execution, event recording, and analytics.

---

## Entry Points

All execution flows through a single function: `aintelope.__main__.run()`.

### `run(config, gui=False)`
- `config`: Either a filename (string, relative to `aintelope/config/`) or a `DictConfig`.
- `gui`: If `True`, launches the experiment GUI before running, and the results viewer after.

### Calling conventions

| Mode | Command | Use case |
|------|---------|----------|
| Default config | `python -m aintelope` | Quick run with defaults |
| Custom config | `python -m aintelope my_config.yaml` | Run a specific experiment |
| GUI | `python -m aintelope --gui` | Configure experiments visually, then run and view results |
| Results viewer | `python -m aintelope --results` | Inspect or revisit results from any previous run |
| Tests | `run("default_config.yaml")` or `run(my_dictconfig)` | Programmatic use from test fixtures |

**Makefile targets:**

| Target | Purpose |
|--------|---------|
| `make venv` | Create a Python 3 virtual environment |
| `make venv-310` | Create a Python 3.10 virtual environment specifically |
| `make clean-venv` | Remove the virtual environment |
| `make install` | Install the project and its gridworld dependencies in editable mode |
| `make install-dev` | Install development tooling (pytest, black, mypy, etc.) |
| `make install-all` | Run both `install` and `install-dev` |
| `make tests-local` | Run fast unit tests — no learning tests, no file output |
| `make tests-learning` | Run learning diagnostics — slow, writes outputs/ |
| `make typecheck-local` | Run mypy type checking |
| `make format` | Apply black code formatter |
| `make format-check` | Check formatting without changing files |
| `make isort` | Sort imports with isort |
| `make isort-check` | Check import order without changing files |
| `make flake8` | Run flake8 linting |
| `make clean` | Remove build and cache artifacts |

**VS Code (`launch.json`):**
Uses `"module": "aintelope"` with args, not `"program"`:
```jsonc
{
    "name": "GUI",
    "type": "debugpy",
    "request": "launch",
    "module": "aintelope",
    "args": ["--gui"]
}
```

**Tests:**
```python
from aintelope.__main__ import run
run("default_config.yaml")   # filename
run(my_dictconfig)           # DictConfig directly
```

### Bootstrap sequence

`run()` handles all bootstrapping in order:
1. `register_resolvers()`: OmegaConf custom resolvers
2. `set_priorities()`: CPU priority (skipped under debugger)
3. `set_memory_limits()`: OS-level memory limits
4. `select_gpu()`: CUDA device selection — runs in a background thread, overlapping with GUI if present. `gpu_thread.join()` gates before experiments begin.

The `if __name__` block only does argument parsing: no setup logic lives there.

---

## Architecture

### Control flow (Mediator Pattern)

The control files orchestrate, the logic modules execute. Control files do not contain domain logic; logic modules do not call each other through the control layer.

```
__main__.run()
  → orchestrator.run_experiments(config)
      → orchestrator.run_trial(cfg, main_config, i_trial)   [per trial, via ProcessPoolExecutor]
          → experiments.run_experiment(cfg, ...)              [per experiment block]
              → environment loop: episodes → steps → agent actions → event logging
```

Control files: `__main__.py`, `orchestrator.py`, `experiments.py`.

### Logic modules

| Module | Responsibility |
|--------|---------------|
| `agents/` | Agent implementations and registry |
| `agents/model/` | Component-based agent model: neural networks, memory, reward inference |
| `environments/` | Environment wrappers and registry |
| `analytics/` | Event recording, run discovery, result analysis, diagnostics |
| `config/` | Config loading, OmegaConf resolvers, system setup |
| `gui/` | Experiment config editor, results viewer |
| `utils/` | Seeding, progress reporting, concurrency helpers, ROI |

### Registries

Agents and environments use a registry pattern: a string key in config maps to a class via `get_agent_class()` / `get_env_class()`.

**Agent registry** (`agents/__init__.py`):
```
"main_agent"      → MainAgent
"random_agent"    → RandomAgent
"sb3_ppo_agent"   → PPOAgent
"sb3_dqn_agent"   → DQNAgent
"sb3_a2c_agent"   → A2CAgent
"llm_agent"       → LLMAgent
```

**Environment registry** (`environments/__init__.py`):
```
"savanna-safetygrid-v1" → SavannaWrapper
```

### Data flow

```
Config (yaml)
  → OmegaConf DictConfig
    → orchestrator distributes trials across workers
      → each experiment block produces EventLog + StateLog + learning_df
        → orchestrator assembles results: {block_name: {events, states, learning_df, manifesto, cfg}}
          → analyze(results) runs configured analytics, returns {analytic_name: {block_name: result}}
            → write_results() serializes events/states to CSV per block
            → each analytic writes its own figures/text/CSVs to outputs_dir
              → results viewer reads CSVs for analysis/playback
```

---

## Config System

### Layering and experiment blocks

`default_config.yaml` serves as the base. Experiment configs (e.g., `example_config.yaml`) contain named blocks of overrides (e.g., `train`, `test`, `train_hard`). At runtime, each block is merged on top of the defaults via `OmegaConf.merge()`, and the orchestrator runs blocks sequentially. Agents carry their models forward between blocks: enabling curriculum learning where agents train on progressively harder scenarios and are tested at arbitrary stages.

Configs are saved per-block in a diff-style fashion: only the overrides are stored, not the full resolved config. This keeps the config architecture lean and composable, and lays the groundwork for future automated config generation (grid search, hyperparameter sweeps) where programmatic block construction needs to be straightforward.

### Custom OmegaConf resolvers

Registered in `config_utils.register_resolvers()`:

| Resolver | Purpose |
|----------|---------|
| `${custom_now:FORMAT}` / `${now:FORMAT}` | Timestamp strings |
| `${abs_path:REL}` | Absolute path from project root |
| `${minus_3:VAL}` | Subtract 3 (used for observation radius) |
| `${muldiv:VAL,MUL,DIV}` | Integer multiply-then-divide |
| `${range:START,END}` | Generate integer range |

### `@ui` annotations

Fields in `default_config.yaml` can carry `@ui` annotations in comments. These are parsed by `ui_schema_manager.py` to auto-generate the GUI:

```yaml
learning_rate: 0.001  # @ui float 0.0001 0.1
agent_class: main_agent  # @ui str main_agent,random_agent,sb3_ppo_agent
test_mode: false      # @ui bool
map_max: 1            # (no annotation → locked/read-only in GUI)
```

### Analytics config block

Under `run:`, the `analytics:` block is a dict of analytic names to parameter dicts. Each key names a library function in `analytics/analytics.py`; the value is passed as `params` to that function. An empty dict `{}` means the analytic runs with no configurable parameters.

```yaml
run:
  write_outputs: True
  analytics:
    run_summary: {}
    learning_improvement:
      episode_fraction: 0.15
      min_improvement_ratio: 1.3
    learning_curve: {}
    loss_curve: {}
    epsilon_curve: {}
    reward_curve: {}
    steps_to_reward: {}
    optimal_efficiency:
      min_efficiency_pct: 0.70
    efficiency_curve: {}
```

Adding or removing an analytic from a run requires only a config change — no code modification. Per-block test overrides follow the same config merge path as all other parameters.

---

## Test Fixtures

### Test suites

Two separate test suites with different purposes:

**`tests/`** — fast unit tests, collected by default pytest sweep and CI:
- `write_outputs: False` — no filesystem side effects
- Runs in seconds per test
- Entry point: `make tests-local`

**`tests/learning/`** — learning diagnostics, excluded from default sweep and CI:
- `write_outputs: True` — writes `outputs/` for post-run inspection
- Runs in minutes per test
- Entry point: `make tests-learning`

### `base_test_config` / `base_learning_config`

Each suite has its own base fixture in its own `conftest.py`. Both provide a minimal single-block config diff on top of `default_config.yaml`. The only meaningful difference is `write_outputs`. Each test merges its own episode count, architecture, and env params on top of the base fixture — two config layers total, no third layer.

### Return value from `run()`

`run()` returns a dict with a `results` key (per-block raw data) and an `analytics` key (per-analytic, per-block computed results):

```python
result = run(cfg)
assert_learning_improvement(result["analytics"]["learning_improvement"]["train"])
report_optimal_policy(result["analytics"]["optimal_efficiency"]["test"])
```

The block names (`"train"`, `"test"`) are the same keys defined in the config. Analytics always return per-block dicts, so tests are explicit about which block they are asserting on — no implicit phase defaulting.

---

## Analytics

### Architecture

`analyze(results)` is the single entry point. It receives the full `results` dict (keyed by block name), iterates `cfg.run.analytics`, and calls each configured library function. It returns `{analytic_name: {block_name: result}}`.

```python
# analyze() internals:
analytics = {}
for name, params in cfg.run.analytics.items():
    analytics[name] = _ANALYTICS[name](results, params)
return analytics
```

Each library function receives the complete `results` dict and its params, computes its analytic independently across all blocks, writes its own outputs (figures, text, CSVs) to `outputs_dir` when `write_outputs` is enabled, and returns computed data keyed by block name. Library functions are standalone — they can be called in isolation from outside `analyze()`.

### Library functions

| Function | Returns per block | Writes |
|----------|------------------|--------|
| `run_summary` | list of summary lines | `report.txt` (appended) |
| `learning_improvement` | `{ratio, start_avg, end_avg, window, passed, min_improvement_ratio}` | `report.txt` |
| `learning_curve` | `{figure}` | `learning_curve.png` |
| `loss_curve` | `{figure}` | `loss_curve.png` |
| `epsilon_curve` | `{figure}` | `epsilon_curve.png` |
| `reward_curve` | `{figure}` | `reward_curve.png` |
| `steps_to_reward` | `{figure}` | `steps_to_reward.png` |
| `optimal_efficiency` | `{efficiency_pct, per_episode, n_episodes, min_efficiency_pct}` | — |
| `efficiency_curve` | `{figure}` | `efficiency_curve.png` |

### Assertion helpers

`assert_learning_improvement(block_result)` and `report_optimal_policy(block_result)` take a single block's result dict directly — the caller selects the block by name:

```python
assert_learning_improvement(result["analytics"]["learning_improvement"]["train"])
report_optimal_policy(result["analytics"]["optimal_efficiency"]["test"])
```

### `DiagnosticsMonitor`

Coordinates resource and learning diagnostics for a single experiment block. Lives in `analytics/diagnostics.py`.

- `sample(label)` — resource snapshot
- `sample_learning(episode, step, report)` — records `loss`, `epsilon`, and `reward` from the agent update report. Skips steps where no gradient update occurred (identified by absence of `loss`).
- `report()` — prints resource report to terminal
- `save_performance(folder)` — writes `performance_report.csv`
- `learning_dataframe()` — returns accumulated `[trial, episode, step, loss, epsilon, reward]` DataFrame

`ResourceMonitor` continues to exist in `utils/performance.py` as the resource-sampling implementation; `DiagnosticsMonitor` owns it as a member.

### Run outputs (gated by `write_outputs`)

When enabled, the run's timestamp folder contains:

| File | Source | Contents |
|------|--------|----------|
| `report.txt` | `run_summary`, `learning_improvement` analytics | Run metadata and improvement summary |
| `learning_curve.png` | `learning_curve` analytic | Per-episode reward per block |
| `loss_curve.png` | `loss_curve` analytic | Per-episode mean loss per block |
| `epsilon_curve.png` | `epsilon_curve` analytic | Epsilon decay per block |
| `reward_curve.png` | `reward_curve` analytic | Per-update reward signal per block |
| `steps_to_reward.png` | `steps_to_reward` analytic | Steps to first reward per block |
| `efficiency_curve.png` | `efficiency_curve` analytic | Per-episode policy efficiency per block |
| `{block}/events.csv` | `write_results()` | Full event log per experiment block |
| `{block}/states.csv` | `write_results()` | Board state per step |
| `{block}/performance_report.csv` | `DiagnosticsMonitor.save_performance()` | Resource snapshots |
| `{block}/env_layouts/{seed}.jpg` | `experiment.py` + `renderer.py` | One image per unique env layout seed used during training |

### Environment layout images

At the end of each experiment block, when `write_outputs` is enabled, a JPEG is rendered for each unique `env_layout_seed` that was used during training. The filename is the seed value itself. Images are written to `{outputs_dir}/{block}/env_layouts/`. The seed set is collected as a natural by-product of the episode loop — no recalculation.

---

## Design Choices

*Each entry documents a deliberate architectural decision, its justification, and any special permissions it grants to break the standard design patterns.*

### 1. Concurrency model

Trials run through a single `ProcessPoolExecutor` path. The worker count is controlled by `cfg.run.max_workers`: set to 0 for auto-detection, or any positive integer to cap manually. Serial execution is simply `max_workers=1`: no separate code path.

Auto-detection (`find_workers`) picks the minimum of available CPUs (or GPU count if CUDA is present) and memory headroom. GPU selection rotates across available devices via a shared SQLite counter so that concurrent launches balance naturally, including across separate processes.

The intent is that a basic user never has to think about hardware utilization. If edge cases arise (e.g., mixed CPU/GPU workloads, cloud-specific constraints), the system can be extended: but the single-variable, single-path design stays.

### 2. Sequential and simultaneous environment modes

Multi-agent environments can follow two execution models: sequential (agents act one at a time, observing each other's intermediate effects) or simultaneous (all agents commit actions before the environment advances). The active mode is set by `cfg.env_params.mode` and branched cleanly in `experiments.py`.

### 3. SB3 baseline training path (special permission)

Stable Baselines 3 agents (`sb3_*`) use an alternate training path. When an SB3 agent is training, control is handed to SB3's own `.learn()` loop, bypassing the main episode/step loop entirely. During test mode, SB3 agents rejoin the standard path. SB3 agent initialization and training are isolated into `_init_sb3_agents()` and `_run_sb3_training()` in `experiments.py`.

**Why:** Classical RL frameworks assume ownership of the training loop: the agent drives environment interaction internally. This is architecturally incompatible with our mediator pattern where the control file owns the loop. Wrapping SB3's loop to match ours would require reimplementing significant framework internals for no functional gain.

**Special permission:** This is an accepted parallel execution path. It exists to provide a credible RL baseline while the project develops its own agent implementations. The expectation is that this path will be deprecated once native agents replace the SB3 dependency.

### 4. Agent and environment registries (factory pattern)

Agents and environments are instantiated via string-keyed registries (`AGENT_REGISTRY`, `ENV_REGISTRY`). A config value like `agent_class: "main_agent"` maps to a class through `get_agent_class()`, and `env: "savanna-safetygrid-v1"` maps through `get_env_class()`. Adding a new agent or environment means implementing the interface, registering it with one line, and making it available in config: no control file changes needed.

This is the mechanism that makes the system agent- and environment-agnostic. The orchestrator does not know or care what it is running, only that it conforms to the expected interface.

### 5. Event-based recording

All experiment runs produce an event log: one row per agent per step, capturing state, action, reward dimensions, and serialized observations. `EventLog` accumulates rows during the experiment, converts to a DataFrame, and is written to CSV per run via `write_results()`.

**Purpose:** Event-level granularity enables analytics beyond mean-reward curves: constraint violation detection, hidden performance scores, and (future) macro-event detection, where event detectors can identify predefined multi-step behavioral patterns from sequences of atomic events.

**Format:** CSV was chosen for accessibility to scientists working with standard data tools. State columns (observations, board state) are compressed and base64-encoded to keep file sizes manageable while preserving the full state needed for visualization and future analytical replay.

### 6. Results viewer variables (special permission)

Variables controlling analytical output: plot ranges, grouping, display parameters: live in the results viewer GUI, not in the experiment config. This is a deliberate exception to the config-driven pattern.

**Why:** These are exploration tools, not experiment parameters. End-users need to fiddle with them interactively while inspecting results: adjusting axis ranges, selecting episodes, changing groupings: until the visualization tells the story they need. Baking these into config would force a re-run cycle for what is fundamentally a post-hoc activity.

**Scope:** The results viewer provides predefined plot libraries that group and present event data in standard ways. The experimenter controls the view parameters; the plot logic itself is in code and can be extended when new analytical needs arise.

### 7. `@ui` config annotations

GUI widget types, ranges, and choices are declared as `@ui` comments directly in `default_config.yaml`, parsed at runtime by `ui_schema_manager.py`. Fields without annotations render as read-only. This keeps the variable definition and its GUI specification in the same file, so there is no separate schema to maintain: adding or changing a config field and its GUI behavior is a single edit in one place.

### 8. Environment demo scripts (legacy)

The scripts under `environments/demos/gridworlds/` launch individual environments in interactive curses mode, bypassing the orchestrator. Their purpose is to let scientists and end-users familiarize themselves with an environment before running agents in it.

### 9. Code archiving

On each run, the current source of both `aintelope/` and `ai_safety_gridworlds/` is zipped into the outputs directory. This is a reproducibility measure: the exact code that produced a set of results is always bundled alongside them.

### 10. Agent-owned models

Each agent owns its Model instance directly. Models handle PyTorch internally: device resolution, neural networks, memory buffers, optimization. Instance lifecycle handles cleanup naturally — when agents go out of scope between experiment blocks, their models are garbage collected. Torch is imported only inside the `agents/model/` package and the bootstrap GPU preload in `config_utils`.

### 11. ROI (Region of Interest)

ROI appends per-agent boolean attention masks to the vision component of each agent's observation. It currently lives as a separate layer-addition step within `experiments.py`, applied after each environment step. The intent is to move ROI into the agents as an internal attention mechanism.

Because SB3's training loop bypasses `experiments.py`, the savanna wrapper applies ROI directly inside its PettingZoo-compatible `step()` method — ensuring SB3 agents receive ROI-augmented observations through their own training path.

### 12. Savanna wrapper

`savanna_wrapper.py` is the interface that allows the legacy savanna gridworld environment to work with the system's `AbstractEnv` contract. It is the sole file that imports from `savanna_safetygrid`. It translates savanna's PettingZoo-derived interface into the canonical MDP contract: dict-format observations `{"vision": ndarray, "interoception": ndarray}`, augmented infos with position and direction data, and a manifesto built on each reset. Legacy-format observations and PettingZoo compatibility are maintained through the `step()` method for SB3 agents.

### 13. Reward inference

In canonical RL, reward is an external signal provided by the environment. This system takes a different approach: reward inference resides within the agent's brain. In nature, organisms infer reward from their sensory observations — they are not handed a scalar by the world. The agent's `RewardInference` component evaluates the observation (after the transition, via `post_activate`) and produces an internal reward signal.

This is also the reason the environment manifesto exists as a feature. The manifesto informs the reward inference about what entities are present in the environment (e.g., which observation layer corresponds to food), so rewards can be inferred correctly regardless of how any particular environment presents its observations. The reward logic is hand-crafted per environment — necessarily so, since what constitutes reward depends on the environment's dynamics — but the manifesto makes the component agnostic to observation format.

### 14. Abstract contracts

Agents conform to `AbstractAgent`: `reset(state, **kwargs)`, `get_action(observation, **kwargs)`, `update(observation, **kwargs)`, `save_model(path, **kwargs)`. The `**kwargs` pattern allows the orchestrator to pass context (step, episode, trial, info, done) uniformly without the abstract class prescribing what each agent needs.

Environments conform to `AbstractEnv`: `reset(**kwargs)`, `step_parallel(actions)`, `step_sequential(actions)`, `manifesto` property, `board_state()`, `score_dimensions` property. The contract is a minimal multi-agent MDP interface. It does not prescribe agent enumeration, observation spaces, or action spaces — that information lives in the manifesto and config.

## Architectural state

SB3 baselines and the savanna gridworld environment are legacy: maintained for validation, but the system is moving toward new agents and environments. Agents and environments both conform to abstract contracts, making the system agnostic to their implementation. Observations follow a canonical dict format. The environment manifesto pattern provides agents with the structural information they need to self-configure.


### 15. Component connectome

The agent's `Model` class manages a **connectome**: a dict of named components that form a directed acyclic graph (DAG) for both activation and learning.

#### Design choices

The connectome is modeled after PyTorch's module composition: components can be arbitrarily chained, stacked, and nested. A NeuralNet component can take observation fields as input, or the output of other components, or both. There is no distinction at the framework level between "observation processors", "planners", or "value estimators" — they are all components with inputs and outputs. The architecture is defined entirely in config.

This means non-sensical architectures can be constructed. Validating cognitive plausibility is the scientist's responsibility, not the framework's. The framework guarantees only that the DAG activates correctly and that learning propagates.

#### Config format

Each agent's `architecture` is a dict of named components. Two keys are reserved: `action` (the activation root, called during `get_action`) and `reward` (called during `update` to compute the reward signal). All other keys are internal components reached through the DAG.

Each entry has two fields:

- `type`: references a library card in the `models:` config section, which defines the Python class, network layers, optimizer, loss function, and other parameters.
- `inputs`: list of names. If a name matches a sibling key in the architecture, it's a component reference. If it matches `observation`, it expands to the environment's modality list from the manifesto. Otherwise, it's an observation field name.

```yaml
architecture:
  action:
    type: DQN
    inputs: [q_net]
  reward:
    type: RewardInference
    inputs: [observation]
  q_net:
    type: DQN-NN
    inputs: [observation]
```

The `observation` keyword is expanded at init time via the environment manifesto's `observation_shapes` keys. This is the single source of truth for available modalities.

#### Activation (pull-based, top-down)

`Model.get_action(obs)` seeds the shared `activations` dict with observation data, then calls `components["action"].activate(activations)`. The action component pulls from its declared inputs: if an input is another component, it calls that component's `activate` first. This recurses to leaf components, which read observation fields from activations. Each component writes its output to `activations[self.component_id]`.

There is no iteration over components in Model during activation. The DAG traversal is implicit in the recursive pull. Components that are never reached from the action root are never activated during `get_action`.

Strategy components like DQN or MCTS may call their input components multiple times with hypothetical states. These calls use temporary dicts to avoid polluting the shared activations namespace.

#### Reward and `done`

`Model.update(next_obs, done)` adds the next observation to activations (prefixed with `next_`), injects `done` as a plain float, then calls `components["reward"].activate(activations)`. The reward component computes an internal reward signal from observation state. `done` is passed through the `**kwargs` chain from `experiment.py` → `MainAgent.update()` → `Model.update()` — the same pattern as other step-context fields (episode, step, trial).

The reward signal and `done` enter learning through shared memory. `done` is used by strategy components to correctly mask terminal states in the Bellman equation.

#### Learning (push-based, top-down via signals)

After reward activation, `Model` pushes the full transition to shared memory, then calls `components["action"].update(signals)` where `signals` is an initially empty dict. The learning DAG mirrors the activation DAG exactly: strategy components propagate training signals down to their subnetworks through `signals`.

**Strategy components** (e.g., `DQN`) own the training logic for their subnetworks. `DQN.update(signals)` computes a Bellman loss closure and writes it into `signals[q_net_id]`, then calls `q_net.update(signals)`. The closure captures the RL algorithm's logic — action indexing, discount factor, target network bootstrap — but is agnostic to the network's shape and device.

**NeuralNet components** own their own training mechanics: batch sampling from shared memory, tensor conversion, forward passes, optimizer step. When a custom loss closure is present in `signals`, `NeuralNet` executes it against its own batch. When no closure is present, `NeuralNet` falls back to self-supervised training using the targets declared in its library card.

This design keeps responsibilities cleanly separated: strategy components define *what* to optimize, NeuralNet defines *how* to optimize. A strategy component can be swapped for a different RL algorithm without touching the NeuralNet implementation, and vice versa.

#### Shared state

- `activations`: flat dict, step-scoped. Populated with observation data at `get_action`, cleared after `update`. Components read inputs from it and write outputs to it.
- `signals`: flat dict, update-scoped. Passed top-down through `update()` calls. Strategy components write loss closures into it; NeuralNets read from it.
- `components`: the persistent dict of component objects. Components can access siblings through this dict (received at init via context).
- `memory`: shared replay memory. All components read from it during learning. The field list is derived from observation shapes, architecture keys, and `done`.

#### Library cards (`models:` config section)

Each component type has a library card defining its class, parameters, and (for NeuralNets) layer architecture. The `type` field in the architecture entry references a key in this section. Class resolution uses the naming convention: names containing `-NN` map to `NeuralNet`; others map to the class matching the type name.

Library cards are defined separately from the architecture so that multiple components can share the same card (e.g. two NeuralNets with identical topology but different roles), and so that scientists can develop new cards without modifying the connectome framework.

#### Component contract

All components implement the `Component` ABC:

- `activate(activations: dict) -> None`: compute output from inputs in activations dict, write result to `activations[self.component_id]`.
- `update(signals: dict = None) -> report or None`: propagate learning. Strategy components write loss closures into signals and call subcomponent `update`. NeuralNets execute training. Non-learning components return None.

Components receive a `context` dict at init containing: `cfg`, `device`, `components`, `memory`, `activations`, `env_manifesto`, `agent_id`, `component_id`, `inputs` (expanded), and `plans` (the library card).

<!-- Entries to be added step by step -->