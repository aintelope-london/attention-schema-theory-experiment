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

You can keep or remove the individual code blocks below it as you see fit — the table covers the same info more concisely.

**CLI (default config):**
```
python -m aintelope
```

**CLI (custom config):**
```
python -m aintelope my_config.yaml
```

**CLI (GUI):**
```
python -m aintelope --gui
```

**CLI (results viewer standalone):**
```
python -m aintelope --results
```

**Makefile targets:**

| Target | Purpose |
|--------|---------|
| `make venv` | Create a Python 3 virtual environment |
| `make venv-310` | Create a Python 3.10 virtual environment specifically |
| `make clean-venv` | Remove the virtual environment |
| `make install` | Install the project and its gridworld dependencies in editable mode |
| `make install-dev` | Install development tooling (pytest, black, mypy, etc.) |
| `make install-all` | Run both `install` and `install-dev` |
| `make tests-local` | Run the test suite with coverage |
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
4. `select_gpu()`: CUDA device selection

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
| `environments/` | Environment wrappers and registry |
| `training/` | PyTorch training utilities (Trainer, models) |
| `analytics/` | Event recording, run discovery, result analysis |
| `config/` | Config loading, OmegaConf resolvers, system setup |
| `gui/` | Experiment config editor, results viewer |
| `utils/` | Seeding, progress reporting, concurrency helpers |

### Registries

Agents and environments use a registry pattern: a string key in config maps to a class via `get_agent_class()` / `get_env_class()`.

**Agent registry** (`agents/__init__.py`):
```
"random_agent"    → RandomAgent
"q_agent"         → QAgent
"sb3_ppo_agent"   → PPOAgent
"sb3_dqn_agent"   → DQNAgent
"sb3_a2c_agent"   → A2CAgent
"llm_agent"       → LLMAgent
```

**Environment registry** (`environments/__init__.py`):
```
"savanna-safetygrid-sequential-v1" → SavannaGridworldSequentialEnv
"savanna-safetygrid-parallel-v1"   → SavannaGridworldParallelEnv
```

### Data flow

```
Config (yaml)
  → OmegaConf DictConfig
    → orchestrator distributes trials across workers
      → each experiment produces an EventLog
        → EventLog.to_dataframe() → pandas DataFrame
          → write_results() serializes to CSV per run
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
agent_class: q_agent  # @ui str random_agent,q_agent,sb3_ppo_agent
test_mode: false      # @ui bool
map_max: 1            # (no annotation → locked/read-only in GUI)
```

---

## Test Fixtures

### `base_test_config`
Loads `default_config.yaml` and applies test-specific overrides for fast test execution.

### `dqn_learning_config`
Extends `base_test_config` for ML learning verification tests. Uses slightly longer runs to allow measurable learning signal.

---

## Design Choices

*Each entry documents a deliberate architectural decision, its justification, and any special permissions it grants to break the standard design patterns.*

### 1. Concurrency model

Trials run through a single `ProcessPoolExecutor` path. The worker count is controlled by `cfg.run.max_workers`: set to 0 for auto-detection, or any positive integer to cap manually. Serial execution is simply `max_workers=1`: no separate code path.

Auto-detection (`find_workers`) picks the minimum of available CPUs (or GPU count if CUDA is present) and memory headroom. GPU selection rotates across available devices via a shared SQLite counter so that concurrent launches balance naturally, including across separate processes.

The intent is that a basic user never has to think about hardware utilization. If edge cases arise (e.g., mixed CPU/GPU workloads, cloud-specific constraints), the system can be extended: but the single-variable, single-path design stays.

### 2. Sequential and simultaneous environment modes

Multi-agent environments can follow two execution models: sequential (agents act one at a time, observing each other's intermediate effects) or simultaneous (all agents commit actions before the environment advances). These correspond to PettingZoo's AEC and Parallel APIs respectively.

The system supports both modes. The active mode is determined by the environment class selected in config. Currently the branching between the two lives as `isinstance` checks in `experiments.py`; this is a known refactor target to be unified behind a common interface.

### 3. SB3 baseline training path (special permission)

Stable Baselines 3 agents (`sb3_*`) use an alternate training path: `run_baseline_training()` in `experiments.py`. When an SB3 agent is training, control is handed to SB3's own `.learn()` loop, bypassing the main episode/step loop entirely. During test mode, SB3 agents rejoin the standard path.

**Why:** Classical RL frameworks assume ownership of the training loop: the agent drives environment interaction internally. This is architecturally incompatible with our mediator pattern where the control file owns the loop. Wrapping SB3's loop to match ours would require reimplementing significant framework internals for no functional gain.

**Special permission:** This is an accepted parallel execution path. It exists to provide a credible RL baseline while the project develops its own agent implementations. The expectation is that this path will be deprecated once native agents replace the SB3 dependency.

### 4. Agent and environment registries (factory pattern)

Agents and environments are instantiated via string-keyed registries (`AGENT_REGISTRY`, `ENV_REGISTRY`). A config value like `agent_class: "q_agent"` maps to a class through `get_agent_class()`, and `env: "savanna-safetygrid-parallel-v1"` maps through `get_env_class()`. Adding a new agent or environment means implementing the interface, registering it with one line, and making it available in config: no control file changes needed.

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

### 10. Shared Trainer instance

Non-SB3 agents within an experiment share a single `Trainer` object, which owns the PyTorch models and optimization. This centralizes device management and model coordination. The current Trainer implementation is legacy and largely stubbed out, but the pattern is intentional: future native agents will use a centralized trainer for shared infrastructure (device placement, checkpointing, optimization scheduling) without agents needing to manage these concerns individually.


## Architectural state

Current environment and sb3-baselines are going to become legacy, we maintain their support for validation, but move to use new envs and agents in the future. 
The system will support "arbitrary agents and environments", as long as they adhere to an abstract class. Agents have this already, pending for envs.



<!-- Entries to be added step by step -->
