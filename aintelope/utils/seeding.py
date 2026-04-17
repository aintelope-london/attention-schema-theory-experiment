# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.


def set_global_seeds(seed: int) -> None:
    """Set random seeds and enable deterministic ops for reproducibility.

    Seeds random/numpy/torch and enables every torch determinism flag. Flags
    are safe no-ops on CPU. On GPU, CUBLAS_WORKSPACE_CONFIG is effective only
    if set before CUDA initialises — export it from the invoking shell for
    full GPU reproducibility.
    """
    import os
    import random
    import numpy as np
    import torch

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
