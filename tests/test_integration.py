# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Integration test: full orchestrator cycle."""

from aintelope.__main__ import run


def test_integration(base_test_config):
    """Run single-block config through orchestrator without errors."""
    run(base_test_config)
