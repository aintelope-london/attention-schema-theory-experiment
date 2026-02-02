"""Integration test: agent completes full pipeline without errors."""

import pytest
from aintelope.__main__ import run
from tests.conftest import as_pipeline


def test_agent_completes_pipeline(learning_config):
    """Agent runs full train + test cycle without errors."""
    run(as_pipeline(learning_config))


if __name__ == "__main__":  # and os.name == "nt":
    pytest.main([__file__])
