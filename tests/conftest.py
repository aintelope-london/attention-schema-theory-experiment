import os
import pytest
from omegaconf import OmegaConf
from aintelope.config.config_utils import register_resolvers

register_resolvers()


@pytest.fixture
def base_test_config():
    """Minimal single-block config for fast test execution.
    Ready to pass to run(). Tests merge their own overrides on top.
    """
    return OmegaConf.create(
        {
            "test": {
                "run": {
                    "trials": 1,
                    "episodes": 1,
                    "steps": 10,
                    "write_outputs": False,
                    "max_workers": 1,
                },
                "env_params": {
                    "map_max": 5,
                },
            }
        }
    )


@pytest.fixture
def base_env_cfg():
    """Full resolved cfg for direct environment construction in tests."""
    cfg = OmegaConf.load(os.path.join("aintelope", "config", "default_config.yaml"))
    return OmegaConf.merge(cfg, {"env_params": {"num_iters": 10, "map_max": 5}})


@pytest.fixture
def base_env_params(base_env_cfg):
    """Flat env_params dict for tests that need raw params."""
    return dict(base_env_cfg.env_params)
