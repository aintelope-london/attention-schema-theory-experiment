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

### Execution hierarchy

The experiment loop nests three levels: trials, episodes, and steps.

**Trials** provide statistical significance. The agent's model is reinitialised from scratch each trial — same config, different random seed. Multiple trials verify that learning is robust and not an artefact of a lucky initialisation.

**Episodes** are environment resets within a trial. Each episode randomises the environment layout (positions, seed) while the agent's model persists and continues learning. Episodes are the unit of training: one episode = one trajectory from reset to termination or step limit.

**Steps** are individual agent–environment interactions within an episode: observe, act, receive next observation.

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
"dummy_agent"     → DummyAgent
"sb3_agent"       → SB3Agent
```

**Environment registry** (`environments/__init__.py`):
```
"gridworld-v1"          → GridworldEnv
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

`default_config.yaml` serves as the base. Experiment configs (e.g., `example_config.yaml`) contain named blocks of overrides (e.g., `train`, `test`, `train_hard`). At runtime, each block is merged on top of the previous block's resolved config via `OmegaConf.merge()`, and the orchestrator runs blocks sequentially. This means `defaults + block_1 → block_2 → block_3`: each block accumulates from what came before, enabling curriculum learning where early-block parameters carry forward unless explicitly overridden.

Agents carry their models forward between blocks: enabling curriculum learning where agents train on progressively harder scenarios and are tested at arbitrary stages.

Configs are saved per-block in a diff-style fashion: only the overrides are stored, not the full resolved config. This keeps the config architecture lean and composable, and lays the groundwork for future automated config generation (grid search, hyperparameter sweeps) where programmatic block construction needs to be straightforward.

### Models library

`models_library.yaml` is a protected config file (never overwritten by the GUI or experiment saves) that contains two top-level sections:

- `models:` — library cards for all component types (optimizers, loss functions, network layer stacks). These are merged into `cfg.models` at load time and referenced by `type` fields in architecture entries.
- `architectures:` — named connectome topologies. Each entry is a complete `architecture:` dict ready to be injected into an agent.

`agent_0.model` in `default_config.yaml` holds the name of the chosen architecture (e.g., `dqn_fc_roi`). `init_config` resolves this name against the library and injects the corresponding architecture into `cfg.agent_params.agents.agent_0.architecture` before any block overrides are applied. `model.py` is unaware of this indirection — it reads `cfg.agent_params.agents.[agent_id].architecture` and `cfg.models` as always.

One model per experiment set — all blocks in a config share the same architecture. Switching models means choosing a different name in `agent_0.model`; there is no per-block model override.

### Custom OmegaConf resolvers

Registered in `config_utils.register_resolvers()`:

| Resolver | Purpose |
|----------|---------|
| `${custom_now:FORMAT}` / `${now:FORMAT}` | Timestamp strings |
| `${abs_path:REL}` | Absolute path from project root |
| `${muldiv:VAL,MUL,DIV}` | Integer multiply-then-divide |
| `${range:START,END}` | Generate integer range |

### `@ui` annotations

Fields in `default_config.yaml` can carry `@ui` annotations in comments. These are parsed by `ui_schema_manager.py` to auto-generate the GUI:

```yaml
learning_rate: 0.001  # @ui float 0.0001 0.1
agent_class: main_agent  # @ui str main_agent,random_agent,sb3_agent,dummy_agent
test_mode: false      # @ui bool
map_size: 1            # (no annotation → locked/read-only in GUI)
```

Fields without annotations render as read-only. This keeps the variable definition and its GUI specification in the same file, so there is no separate schema to maintain: adding or changing a config field and its GUI behavior is a single edit in one place.

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
    visitation_heatmap:
      n_windows: 2
    action_distribution:
      n_windows: 2
```

Adding or removing an analytic from a run requires only a config change — no code modification. Per-block test overrides follow the same config merge path as all other parameters.

---

## Agent Model

### Code archiving

On each run, the current source of `aintelope/` is zipped into the outputs directory. This is a reproducibility measure: the exact code that produced a set of results is always bundled alongside them.

### Agent-owned models

Each agent owns its Model instance directly. Models handle PyTorch internally: device resolution, neural networks, memory buffers, optimization. Instance lifecycle handles cleanup naturally — when agents go out of scope between experiment blocks, their models are garbage collected. Torch is imported only inside the `agents/model/` package and the bootstrap GPU preload in `config_utils`.

### ROI (Region of Interest)

The ROI component is a vision pre-filter in the agent's connectome DAG. It reads `activations["internal_action"]` from the previous step, updates its internal `roi_state` for each declared feature, computes a boolean mask, darkens `vision[:-1]` outside the mask, writes the mask into `vision[-1]`, and surfaces it in `activations[component_id]` for the environment to blit into `state["mask"]`.

`roi_features` in the architecture entry declares which state dimensions are agent-controllable. Each feature contributes exactly 3 internal actions (neg, stay, pos). `n_internal_actions` is declared explicitly in the yaml as `len(roi_features) * 3` — the config is the authority, no dynamic computation in code.

When ROI is absent from the architecture entirely, the env wrapper still appends a zero channel as `vision[-1]` for consistent observation shapes. `ROI.activate()` is simply never called — no vestigial branches needed.

The environment receives the viewport-space ROI mask from the agent's action dict (`actions[aid]["roi"]`), blits it to absolute board coordinates via `_blit_roi()`, and stores it in `state["mask"]` as an `(N_agents, H, W)` float32 array. This makes each agent's ROI visible to all other agents through the shared state. The results viewer reads `state["mask"]` directly from `states.csv` to render the overlay — no geometry is recomputed at display time.

`roi_state` keys:

| Key | Present when | Driven by |
|---|---|---|
| `angle` | always for `circle` mode | actions if in `roi_features`, else fixed at 0.0 |
| `radius` | `radius` in `roi_features` | actions, clamped to `[circle_radius_min, circle_radius_max]` |
| `distance` | `roi_mode: circle` | fixed, seeded from `circle_distance` |

### Reward inference

In canonical RL, reward is an external signal provided by the environment. This system takes a different approach: reward inference resides within the agent's brain. In nature, organisms infer reward from their sensory observations — they are not handed a scalar by the world. The agent's `RewardInference` component evaluates the observation (after the transition, via `post_activate`) and produces an internal reward signal.

This is also the reason the environment manifesto exists as a feature. The manifesto informs the reward inference about what entities are present in the environment (e.g., which observation layer corresponds to food), so rewards can be inferred correctly regardless of how any particular environment presents its observations. The reward logic is hand-crafted per environment — necessarily so, since what constitutes reward depends on the environment's dynamics — but the manifesto makes the component agnostic to observation format.

### Abstract contracts

Agents conform to `AbstractAgent`: `reset(state, **kwargs)`, `get_action(observation, **kwargs)`, `update(observation, **kwargs)`, `save_model(path, **kwargs)`. The `**kwargs` pattern allows the orchestrator to pass context (step, episode, trial, info, done) uniformly without the abstract class prescribing what each agent needs.

Environments conform to `AbstractEnv`: `reset(**kwargs)`, `step_parallel(actions)`, `step_sequential(actions)`, `manifesto` property, `board_state()`, `score_dimensions` property. The contract is a minimal multi-agent MDP interface. It does not prescribe agent enumeration, observation spaces, or action spaces — that information lives in the manifesto and config.

### Component connectome

The agent's `Model` class manages a **connectome**: a dict of named components that form a directed acyclic graph (DAG) for both activation and learning.

#### Design choices

The connectome is modeled after PyTorch's module composition: components can be arbitrarily chained, stacked, and nested. A NeuralNet component can take observation fields as input, or the output of other components, or both. There is no distinction at the framework level between "observation processors", "planners", or "value estimators" — they are all components with inputs and outputs. The architecture is defined entirely in config.

This means non-sensical architectures can be constructed. Validating cognitive plausibility is the scientist's responsibility, not the framework's. The framework guarantees only that the DAG activates correctly and that learning propagates.

#### Config format

Each agent's `architecture` is a dict of named components. Two keys are reserved: `action` (the activation root, called during `get_action`) and `reward` (called during `update` to compute the reward signal). All other keys are internal components reached through the DAG.

Each entry has two fields:

- `type`: references a library card in `cfg.models` (populated from `models_library.yaml`), which defines the Python class, network layers, optimizer, loss function, and other parameters.
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

#### Library cards (`models_library.yaml` → `cfg.models`)

Each component type has a library card defining its class, parameters, and (for NeuralNets) layer architecture. The `type` field in the architecture entry references a key in `cfg.models`, which is populated at load time from `models_library.yaml`. Class resolution uses the naming convention: names containing `-NN` map to `NeuralNet`; others map to the class matching the type name.

Library cards are defined separately from the architecture so that multiple components can share the same card (e.g. two NeuralNets with identical topology but different roles), and so that scientists can develop new cards without modifying the connectome framework. Adding a new card requires only an entry in `models_library.yaml` — no code changes.

#### Component contract

All components implement the `Component` ABC:

- `activate(activations: dict) -> None`: compute output from inputs in activations dict, write result to `activations[self.component_id]`.
- `update(signals: dict = None) -> report or None`: propagate learning. Strategy components write loss closures into signals and call subcomponent `update`. NeuralNets execute training. Non-learning components return None.

Components receive a `context` dict at init containing: `cfg`, `device`, `components`, `memory`, `activations`, `env_manifesto`, `agent_id`, `component_id`, `inputs` (expanded), and `plans` (the library card).

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

### ROI analytics

Two analytics track whether agents with ROI components are using their attention meaningfully.

**`roi_turn_distribution`**: plots the frequency of each internal action (left/stay/right turn) across episode windows. Reveals whether the agent develops a preferred rotation strategy over training. No-ops for non-ROI runs — `Internal_action` is null for those agents, `dropna` produces an empty DataFrame, and the function returns `{}`.

**`roi_food_alignment`**: plots the rate at which food appears inside the agent's ROI across episode windows. Computed directly from the logged `Observation`: the food channel (`manifesto["food_ind"]`) and the ROI channel (last vision channel, always appended by the env) are ANDed per step. No geometry is re-implemented — the mask the agent actually used is what was logged. Works for any entity present in the observation space (predators, other agents) by channel index; food is the first use case.

Both analytics follow the null-object pattern: absent ROI data flows through as empty and produces no output, with no branching needed.

### DiagnosticsMonitor

Owns resource sampling, learning metrics, and progress reporting:

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
| `visitation_heatmap_{block}.png` | `visitation_heatmap` analytic | Grid visitation count heatmap, early vs late episodes, one file per block |
| `action_distribution_{block}.png` | `action_distribution` analytic | Action frequency bar charts, early vs late episodes, one file per block |
| `{block}/events.csv` | `write_results()` | Full event log per experiment block |
| `{block}/states.csv` | `write_results()` | Board state per step |
| `{block}/performance_report.csv` | `DiagnosticsMonitor.save_performance()` | Resource snapshots |
| `{block}/env_layouts/{seed}.jpg` | `experiment.py` + `renderer.py` | One image per unique env layout seed used during training |

### Environment layout images

At the end of each experiment block, when `write_outputs` is enabled, a JPEG is rendered for each unique `env_layout_seed` that was used during training. The filename is the seed value itself. Images are written to `{outputs_dir}/{block}/env_layouts/`. The seed set is collected as a natural by-product of the episode loop — no recalculation.

---

## Design Choices

*Each entry documents a deliberate architectural decision, its justification, and any special permissions it grants to break the standard design patterns.*

### Concurrency model

Trials run through a single `ProcessPoolExecutor` path. The worker count is controlled by `cfg.run.max_workers`: set to 0 for auto-detection, or any positive integer to cap manually. Serial execution is simply `max_workers=1`: no separate code path.

Auto-detection (`find_workers`) picks the minimum of available CPUs (or GPU count if CUDA is present) and memory headroom. GPU selection rotates across available devices via a shared SQLite counter so that concurrent launches balance naturally, including across separate processes.

The intent is that a basic user never has to think about hardware utilization. If edge cases arise (e.g., mixed CPU/GPU workloads, cloud-specific constraints), the system can be extended: but the single-variable, single-path design stays.

### Sequential and simultaneous environment modes

Multi-agent environments can follow two execution models: sequential (agents act one at a time, observing each other's intermediate effects) or simultaneous (all agents commit actions before the environment advances). The active mode is set by `cfg.env_params.mode` and branched cleanly in `experiments.py`.

### SB3 baseline agents (special permission)

SB3 exists specifically to provide a credible RL reference point for verifying that `MainAgent` is learning correctly — not as a target for further development. All SB3 algorithms (DQN, PPO, A2C) are accessed through a single `SB3Agent` class; the algorithm is selected via the `algorithm` config field.

**Training path (special permission):** When an SB3 agent is training, control is handed to SB3's own `.learn()` loop via `SB3Agent.training()`, bypassing the main episode/step loop entirely. During test mode, SB3 agents rejoin the standard canonical path. This is architecturally incompatible with our mediator pattern where the control file owns the loop, but wrapping SB3's loop to match ours would require reimplementing significant framework internals for no functional gain.

**Reward:** Where `MainAgent` infers reward from observation via `RewardInference`, SB3 reads reward directly from `interoception[0]` inside `GridworldGymWrapper`. This is a consequence of SB3 owning its own training loop and having no access to the agent's brain.

**Analytics on the SB3 training block:** SB3's training path does not participate in the event-logging loop. The train block receives a dummy-row DataFrame so analytics flow through without branching — consistent with the null-object pattern. Results from this block are meaningless and intentionally ignored.

**Special permission:** This is an accepted parallel execution path. It exists to provide a credible RL baseline while the project develops its own agent implementations. The expectation is that this path will be deprecated once native agents fully replace the SB3 dependency.

### Agent and environment registries (factory pattern)

Agents and environments are instantiated via string-keyed registries (`AGENT_REGISTRY`, `ENV_REGISTRY`). A config value like `agent_class: "main_agent"` maps to a class through `get_agent_class()`, and `env: "gridworld-v1"` maps through `get_env_class()`. Adding a new agent or environment means implementing the interface, registering it with one line, and making it available in config: no control file changes needed.

This is the mechanism that makes the system agent- and environment-agnostic. The orchestrator does not know or care what it is running, only that it conforms to the expected interface.

### Event-based recording

All experiment runs produce an event log: one row per agent per step, capturing state, action, reward dimensions, and serialized observations. `EventLog` accumulates rows during the experiment, converts to a DataFrame, and is written to CSV per run via `write_results()`.

**Purpose:** Event-level granularity enables analytics beyond mean-reward curves: constraint violation detection, hidden performance scores, and (future) macro-event detection, where event detectors can identify predefined multi-step behavioral patterns from sequences of atomic events.

**Format:** CSV was chosen for accessibility to scientists working with standard data tools. State columns (observations, board state) are compressed and base64-encoded to keep file sizes manageable while preserving the full state needed for visualization and future analytical replay.

### Results viewer variables (special permission)

Variables controlling analytical output: plot ranges, grouping, display parameters: live in the results viewer GUI, not in the experiment config. This is a deliberate exception to the config-driven pattern.

**Why:** These are exploration tools, not experiment parameters. End-users need to fiddle with them interactively while inspecting results: adjusting axis ranges, selecting episodes, changing groupings: until the visualization tells the story they need. Baking these into config would force a re-run cycle for what is fundamentally a post-hoc activity.

**Scope:** The results viewer provides predefined plot libraries that group and present event data in standard ways. The experimenter controls the view parameters; the plot logic itself is in code and can be extended when new analytical needs arise.

### Test mode gating

The `test_mode` check that prevents weight updates during evaluation sits inside `NeuralNet.optimize()` — not in the orchestration layer or `DQN.update()`. NeuralNet owns its own training mechanics; test mode is a training mechanic concern, not a control flow concern. The gate is a single condition alongside the existing batch-size check:

```python
if self.cfg.run.experiment.test_mode or len(self.memory) < self.cfg.agent_params.batch_size:
    return {}
```

No other file needs to know about test mode gating. `Model.update()`, `DQN.update()`, `memory.push`, and the abstract contract all remain unchanged.

### Memory limits (Linux-only)

`set_memory_limits()` sets `RLIMIT_AS` to available RAM via `psutil` and Python's `resource` module. Windows memory limiting was removed — a prior Job Object approach silently failed. Windows remains a supported platform otherwise.

### Config block cascading

Each block merges onto the previous block's resolved config: `defaults + block_1 → block_2 → block_3`, not `defaults + block_N`. This enables curriculum learning where early-block parameters carry forward unless explicitly overridden. The cascade is implemented by reassigning `cfg` in the block loop inside `run_trial`.

### `wait` action excluded from canonical validation

The `wait` action is configurable but deliberately excluded from foraging validation scenarios. In a foraging task `wait` is never optimal and only widens the random exploration space, inflating the sample budget required to converge. The default action list in `default_config.yaml` does not include `wait`; validation scenarios inherit this.

### Food consumption

When an agent steps onto a ripe food tile, the food disappears permanently: `_food_age` is popped and the agent tile overwrites the cell. No new food is spawned on consumption. The ripening cycle (`unripe → ripe → rotten → bush → unripe → ...`) manages food lifecycle at existing tracked positions only — each food cell cycles in-place, it does not respawn at a random floor cell.

### `update_frequency` (DQN strategy parameter)

The `update_frequency` field lives in the DQN library card (not in NeuralNet). It gates the optimizer step — the target network update runs independently of this gate and ticks on its own schedule. This feature is present in the codebase but currently untested; `update_frequency` is set to `1` (every step) in `model_library.yaml`, making it a no-op.

### Termination modes

Episode termination is controlled by `cfg.env_params.termination`. When set to `"food"`, the episode ends as soon as the agent consumes food (`Done=True`). When set to `null`, the episode runs to its full step budget regardless of food contact. This is a config-driven behavior switch with no branching in the control files — the environment sets `done` in its state dict, and the experiment loop reads it.

### GridworldEnv

Minimal randomised MOMA gridworld.

**Tile layer order** (`LAYERS` is the canonical reference for all channel indexing):

| Index | Name | Notes |
|---|---|---|
| 0 | floor | |
| 1 | wall | |
| 2 | predator | persists after contact |
| 3 | food | ripe food |
| 4 | food_unripe | ripening mode only |
| 5 | food_rotten | ripening mode only |
| 6 | rock | |
| 7 | water | |
| 8 | bush | |
| 9+ | agent_N | one layer per agent in sorted order |

**Ripening cycle** (active when `env_params.ripening > 0`): each food cell advances through stages (`bush → unripe → ripe → rotten → bush → ...`) every `ripening` steps, cycling in-place at the same board position. Only `food` (ripe) contact fires `interoception[0]` and counts as `ate_food` — the two signals are always identical.

**Interoception channels:** `[0]` food reward this step, `[1]` contact (`+1.0` predator, `-1.0` wall/agent block).

**Observation encoding** is selected by `env_params.observation_format`; add `_encode_<key>()` to extend without touching anything else.

**ASCII map layout:** when `env_params.map_layout` is set, the board is initialised from a multiline string instead of random placement. Character vocabulary: `.` floor, `#` wall, `F` food, `u` food_unripe, `x` food_rotten, `P` predator, `r` rock, `w` water, `b` bush, `A` agent (assigned in sorted order). When `map_layout` is absent, random placement is used with `env_params.map_size` and `env_params.objects`.

### File-based tileset

Tiles are loaded from individual PNG files in `gui/tiles/`. The keyword for each tile is its filename stem (e.g. `food.png` → keyword `"food"`). Adding a new tile requires only dropping a PNG into the directory — no code changes. Draw order is declared in `_DRAW_ORDER` in `renderer.py` and must be updated when new tile types are added. All tiles in the directory must be the same pixel dimensions.

### DummyAgent and animation scripts

`DummyAgent` is a scripted agent that replays a named action sequence from `agents/scripts.py`. It conforms fully to `AbstractAgent` and is registered as `"dummy_agent"` in the agent registry. No model, no learning — `update` and `save_model` are no-ops.

The script name is declared in config under `agent_params.agents.[id].script` and resolves to a list of `{"action": int, ...}` dicts in `scripts.py`. When the sequence is exhausted the agent emits `wait` indefinitely.

If the agent's architecture entry contains an ROI component, `DummyAgent` instantiates it directly (without a full Model) and drives it with `internal_action` from the script each step, returning the resulting mask as `"roi"` in the action dict. This allows ROI animations to be authored as plain integer sequences in `scripts.py` while reusing the ROI component code exactly.

Animation configs live in `animation_config.yaml` as named blocks, following the same block structure as experiment configs. Each block sets `env`, `map_layout` (or `objects`), agent script, and episode/step counts. Animations are run and exported through the standard results viewer export path.

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

Each suite has its own base fixture in its own `conftest.py`. Both provide a minimal single-block config diff on top of `default_config.yaml`. The only meaningful difference is `write_outputs`. Each test merges its own episode count, model, and env params on top of the base fixture — two config layers total, no third layer.

### `tests/learning/test_validation.py` — Canonical validation suite

A locked set of empirical claims about agent capability. Tests here are graduated from `test_lab.py` once a result is reproducible and ready to serve as a standing record.

**Naming convention:** `test_<claim>__<method>__<scope>`
- `claim` — what the test asserts (`foraging`, `generalizes`, etc.)
- `method` — the model or architectural variant (`dqn_fc`, `dqn_cnn`, etc.)
- `scope` — the environment scale (`2x2`, `5x5`, `5x13`, etc.)

**Structure invariants across all tests:**
- `trials: 5`, `train steps: 20`, `test steps: 10` — held constant so differences between tests are attributable to model and scope only
- `greedy_until: 0.3` in train, `greedy_until: 0.0` in test — canonical exploration schedule
- `min_efficiency_pct` is set per-test in the test block's analytics override, never left to the config default
- Single assertion per test: `report_optimal_policy` on the `test` block

**`min_efficiency_pct` policy:**
- `1.0` on 2x2 — this is the system correctness gate. If a DQN-FC cannot achieve 100% on the smallest non-trivial map, there is a bug, not a performance issue. It does not run in production until this passes.
- Lower thresholds on larger scopes encode the expected performance of that model class at that scale, as established empirically.

`tests/learning/test_lab.py` is the development counterpart — no invariants enforced, tests are skipped by default, and anything here is a candidate for eventual graduation into `test_validation.py`.

### Return value from `run()`

`run()` returns a dict with a `results` key (per-block raw data) and an `analytics` key (per-analytic, per-block computed results):

```python
result = run(cfg)
assert_learning_improvement(result["analytics"]["learning_improvement"]["train"])
report_optimal_policy(result["analytics"]["optimal_efficiency"]["test"])
```

The block names (`"train"`, `"test"`) are the same keys defined in the config. Analytics always return per-block dicts, so tests are explicit about which block they are asserting on — no implicit phase defaulting.

### DQN parameter optimization — foraging baseline

This section records the parameter search that established the canonical DQN-FC and DQN-CNN training config used in `test_validation.py`.

#### Baseline parameters (pre-optimization)

| param | dqn_fc | dqn_cnn |
|---|---|---|
| `episodes` | 10 000 | 10 000 |
| `batch_size` | 200 | 550 |
| `replay_buffer_size` | 7 000 | 30 000 |
| `gamma` | 0.99 | 0.99 |
| `greedy_until` (train) | 0.3 | 0.3 |

Results (5 trials, `test_mode=True`, 500 test episodes):

| test | dqn_fc | dqn_cnn |
|---|---|---|
| `test_foraging_2x2` | 97.4% | 97.1% |
| `test_foraging_5x5` | 83.0% | 81.5% |
| `test_generalizes_5x5_to_13x13` | 4.1% | 5.8% |

#### Optimized parameters (current)

Sourced from `test_lab.py` (`test_foraging_dqn_fc_5x5`). The same parameter set was applied to both model variants and all scopes, and yielded substantial gains across the board — including the generalization test which was not the target of the search.

| param | dqn_fc | dqn_cnn |
|---|---|---|
| `episodes` | 7 500 | 7 500 |
| `batch_size` | 350 | 350 |
| `replay_buffer_size` | 30 000 | 30 000 |
| `gamma` | 0.99 | 0.99 |
| `greedy_until` (train) | 0.3 | 0.3 |

Results:

| test | dqn_fc | dqn_cnn |
|---|---|---|
| `test_foraging_2x2` | 92.9% | — |
| `test_foraging_5x5` | 92.9% | 95.2% |
| `test_generalizes_5x5_to_13x13` | 94.2% | 93.7% |

**Note:** `test_foraging_2x2[dqn_fc]` regressed from 97.4% to 92.9% under the new params. The 2x2 scenario is the system correctness gate (`min_efficiency_pct: 1.0`) and is not yet passing. Active investigation ongoing — primary hypothesis is that the `wait` action has been added to the default action space since the original 100% run, inflating the random exploration budget needed for this trivially small map.

---

## Cloud setup

Experiments run on cloud GPU instances via `cloud.sh`, a bootstrap script at the repo root. It is provider-agnostic and has been verified on Lambda Labs. Avoid provider-managed ML stacks (e.g. Lambda Stack) — `install.py` manages all dependencies explicitly and pre-installed frameworks risk version conflicts. The recommended instance image is one that pre-installs CUDA/cuDNN/NCCL at the system level while leaving the Python environment unmanaged, so `install.py` retains full control over Python dependencies.

A single A100 instance is sufficient for the current workload. Multi-node clusters are out of scope.

Tested providers: Lambda Labs. RunPod is a viable alternative if Lambda has no capacity.

The per-instance workflow, SSH key setup, tmux session management, and result retrieval commands are in `README.md`.

---

### Determinism invariants

Bit-exact reproducibility per seed is required for locked-value tests
(`tests/test_learning_smoke.py`). Three mechanisms combine to guarantee it:

**`set_global_seeds(seed)`** (`aintelope/utils/seeding.py`) seeds
`random`/`numpy`/`torch` and enables every torch determinism flag:

- `torch.backends.cudnn.deterministic = True`
- `torch.backends.cudnn.benchmark = False`
- `torch.use_deterministic_algorithms(True)`
- `os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"`

These are safe no-ops on CPU. On GPU, the CUBLAS var is only effective if set
before CUDA initialises — for GPU reproducibility work, export it from the
invoking shell rather than relying on `set_global_seeds` to pick it up in time.
`torch.use_deterministic_algorithms(True)` does **not** remove stochasticity from
the agent's experience (that comes from seeded RNG streams); it only removes
algorithmic nondeterminism. Statistical robustness is still obtained by varying
`cfg.run.seed` across trials, and each trial remains bit-reproducible.

**`PYTHONHASHSEED=0`** is exported at the top of the Makefile, so both
`tests-local` and `tests-validation` inherit it. GUI launches and IDE debugger
invocations do not — this is an accepted gap because the codebase is held free
of hash-dependent numerics by the next mechanism.

**`tests/test_determinism_invariants.py`** is an AST tripwire that walks every
`.py` file under `aintelope/` and `tests/` and fails if it finds any of:

- a call to `hash(...)`
- iteration over a set or frozenset literal
- iteration over `set(...)` or `frozenset(...)`
- iteration over a module- or function-local name bound to any of the above

The message on failure is `"breaks PYTHONHASHSEED determinism, it is time."` —
the tripwire exists so the day determinism becomes load-bearing beyond the test
suite is the day it is noticed, not silently later. Known limitation: iteration
over attribute-held sets (e.g. `self._cells = set(); for c in self._cells`) is
not caught statically; the smoke test catches any resulting numeric drift as a
second line of defence.

### Learning smoke test

`tests/test_learning_smoke.py` runs one 1500-episode training trial on the 2x2
foraging task (`dqn_fc`, seed 0) and asserts three tier-locked numeric values
against the fixed-seed result: early-window mean reward (eps 200–300),
mid-window mean reward (eps 1200–1500), and test-block `efficiency_pct`. Any
code change that affects learning produces numeric drift; the three tiers
isolate *when* in training the change first shows up, which narrows the search
when something breaks.

The smoke test is a separate concern from `tests/validation/test_validation.py`:
validation tests encode empirical performance claims (threshold-based, relaxed
as parameter searches shift the frontier), while the smoke test locks
reproducibility (exact-match, moves only on intentional relock). The two tests
share neither config nor fixtures. Shrunk 2x2 budget, `dqn_fc`, and
`greedy_until=0.5` are chosen so the run completes in seconds on a CPU and so
tier 2's window observes a fully-greedy policy.

Relock procedure: when a tier fails with an upward delta, review the change,
run the smoke test once more in isolation, and paste the observed value into
the corresponding `_TIER{N}_*` constant at the top of the file.


---

## Attribution & License

Original upstream repository: https://github.com/aintelope-london/attention-schema-theory-experiment

License: see `LICENSE.txt`.
Code contribution details: see `AUTHORS.md`.
