import psutil
import os
import torch

ESTIMATED_MB_PER_WORKER = 512  # PyTorch + SB3 + env baseline


def find_workers(max_workers: int, num_trials: int) -> int:
    """Resolve worker count. 0 = auto-detect from hardware."""
    cpu_cap = torch.cuda.device_count() or max(os.cpu_count() - 1, 1)
    mem_cap = psutil.virtual_memory().available // (512 * 1024 * 1024)
    auto = min(cpu_cap, mem_cap)
    effective = max_workers or auto
    return max(min(effective, num_trials), 1)
