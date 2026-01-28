"""Integration test: agent completes full pipeline without errors."""

import sys
import os
import pytest
from aintelope.pipeline import aintelope_main


def test_agent_completes_pipeline():
    """Agent runs full train + test cycle without errors."""
    os.environ["PIPELINE_CONFIG"] = "config_tests.yaml"
    sys.argv = sys.argv[:1] 
    aintelope_main()

if __name__ == "__main__": #and os.name == "nt":  
    pytest.main([__file__])  # run tests only in this file
