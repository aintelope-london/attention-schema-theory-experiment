"""Integration test: full orchestrator cycle."""

from aintelope.__main__ import run


def test_integration(base_test_config):
    """Run single-block config through orchestrator without errors."""
    run(base_test_config)
