import os
import random

import numpy as np
import torch


def seed_everything(seed):
    """Sets seed for all random operations, ensures reproducibility.

    Adapted from: https://github.com/qhd1996/seed-everything
    """

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # NOTE: Uncommenting the line below may require setting CUDA environment variables before running the program
    #   (for CUDA >= 10.2: CUBLAS_WORKSPACE_CONFIG=:4096:8 or CUBLAS_WORKSPACE_CONFIG=:16:8
    #   See: https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility)
    # torch.use_deterministic_algorithms(True)


def parse_cfg_str(inp, casttype):
    if casttype is None:
        return None if inp == "None" else inp
    else:
        return casttype(inp) if isinstance(inp, str) else inp


def get_size_per_class(data, num_classes=10):
    size = torch.tensor([0] * num_classes, dtype=torch.float32)

    for feature, label in data:
        size[label] += 1

    return size
