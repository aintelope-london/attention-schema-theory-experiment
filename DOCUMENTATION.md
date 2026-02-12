## Entrypoint (`__main__.py`)

All execution flows through a single function: `aintelope.__main__.run()`.

### `run(config, gui=False)`
- `config`: Either a filename (string, relative to `aintelope/config/`) or a `DictConfig`.
- `gui`: If `True`, launches the GUI before running experiments.

### Three calling conventions

**CLI (default config):**
```
python -m aintelope
```

**CLI (custom config):**
```
python -m aintelope config_custom.yaml
```

**CLI (GUI):**
```
python -m aintelope --gui
```

**Tests:**
```python
from aintelope.__main__ import run

# With a config filename
run("config_tests.yaml")

# With a DictConfig (e.g. from a fixture)
run(base_test_config)
```

### VS Code (`launch.json`)
Use `"module": "aintelope"` instead of `"program"`:
```jsonc
{
    "name": "Run orchestrator",
    "type": "debugpy",
    "request": "launch",
    "module": "aintelope",
    "args": ["config_custom.yaml"]
}
```

### Bootstrap sequence
`run()` handles all bootstrapping in order: `register_resolvers()`, `set_priorities()`, `set_memory_limits()`, `select_gpu()`. The `if __name__` block only does argument parsing — no setup logic lives there.

---

## Test Fixtures (conftest.py)

### `base_test_config`
Loads `config_experiment.yaml` and applies minimal overrides for fast test execution.

**Why not hardcoded?** The external `ai_safety_gridworlds` environment expects certain parameters (e.g., `warm_start_steps`) to exist. Using the full config as a base avoids duplicating legacy params across test files.

**Overrides applied:**
- `episodes` capped at 5
- `num_iters` capped at 50
- `warm_start_steps` capped at 10

### `dqn_learning_config`
Extends `base_test_config` for ML learning verification tests. Uses slightly longer runs (50 episodes, 100 iters) to allow measurable learning signal.

**Future:** Agent-specific fixtures (e.g., model-based agents) should live in their respective test files, not here.