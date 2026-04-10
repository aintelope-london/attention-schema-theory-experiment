# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""
Development sandbox. Tests here are works in progress — not yet validated, not part of the canonical evidence record. Graduate to test_validation.py once a test is reproducibly passing and the claim is ready to be locked.
"""

import pytest
from omegaconf import OmegaConf

from aintelope.__main__ import run
from aintelope.analytics.analytics import (
    assert_learning_improvement,
    report_optimal_policy,
)


@pytest.mark.skip(reason="ModelBased debugging pending")
def test_model_based_learns(base_learning_config):
    """main_agent with ModelBased architecture shows learning improvement."""
    cfg = OmegaConf.merge(
        base_learning_config,
        {
            "train": {
                "run": {
                    "experiment": {
                        "steps": 11,
                        "episodes": 500,
                    },
                },
                "agent_params": {
                    "roi_mode": None,
                    "agents": {
                        "agent_0": {
                            "architecture": {
                                "action": {
                                    "type": "ModelBased",
                                    "inputs": ["dynamic", "value"],
                                },
                                "reward": {
                                    "type": "RewardInference",
                                    "inputs": ["observation"],
                                },
                                "dynamic": {
                                    "type": "NextState-NN",
                                    "inputs": ["observation"],
                                },
                                "value": {
                                    "type": "StateValue-NN",
                                    "inputs": ["observation"],
                                },
                            },
                        },
                    },
                },
                "env_params": {},
            },
        },
    )
    result = run(cfg)
    assert_learning_improvement(result["analytics"]["learning_improvement"]["train"])


if __name__ == "__main__":
    pytest.main([__file__])
