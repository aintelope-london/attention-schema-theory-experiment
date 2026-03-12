# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import os
import pytest
from omegaconf import OmegaConf


@pytest.fixture
def base_learning_config():
    """Minimal single-block config for learning tests.
    write_outputs=True so diagnostics and reports are written to outputs/.
    """
    return OmegaConf.create(
        {
            "train": {
                "run": {
                    "trials": 1,
                    "write_outputs": True,
                    "max_workers": 1,
                    "experiment": {
                        "episodes": 1,
                        "steps": 10,
                    },
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
