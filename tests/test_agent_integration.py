"""Integration test: agent completes full pipeline without errors."""

import sys
import os
import pytest
from aintelope.__main__ import run


def test_agent_completes_pipeline():
    """Agent runs full train + test cycle without errors."""
    run("config_tests.yaml")


if __name__ == "__main__":  # and os.name == "nt":
    pytest.main([__file__])  # run tests only in this file
