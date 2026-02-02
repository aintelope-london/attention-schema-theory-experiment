## Test Fixtures (conftest.py)

### `base_test_config`
Loads `default_config.yaml` and applies minimal overrides for fast test execution.

**Why not hardcoded?** The external `ai_safety_gridworlds` environment expects certain parameters (e.g., `warm_start_steps`) to exist. Using the full config as a base avoids duplicating legacy params across test files.

**Overrides applied:**
- `unit_test_mode = True`
- `num_episodes` capped at 5
- `num_iters` capped at 50
- `warm_start_steps` capped at 10


**Future:** Agent-specific fixtures (e.g., model-based agents) should live in their respective test files, not here.