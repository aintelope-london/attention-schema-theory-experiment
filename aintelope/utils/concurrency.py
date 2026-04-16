# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import psutil
import os
import torch

ESTIMATED_MB_PER_WORKER = 2048  # PyTorch + SB3 + env baseline


def find_workers(max_workers: int, num_trials: int) -> int:
    """Resolve worker count. 0 = auto-detect from hardware."""
    cpu_cap = max(os.cpu_count() - 1, 1)
    mem_cap = psutil.virtual_memory().available // (
        ESTIMATED_MB_PER_WORKER * 1024 * 1024
    )
    auto = min(cpu_cap, mem_cap)
    effective = max_workers or auto
    return max(min(effective, num_trials), 1)
