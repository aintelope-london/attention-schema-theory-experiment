def set_global_seeds(seed: int) -> None:
    """Set all random seeds for reproducibility.

    Args:
        seed: Base seed value
    """
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
