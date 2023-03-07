import logging

import torch

# TODO: May just use transformation classes themselves rather than their names in TRANSFORM_NAMES
# from torchvision import transforms as tr

from .utils import seed_everything

_global_seed = None
_logging_level = logging.WARNING
_dtype = None  # Data type to be used in tensors

logging.basicConfig(
    format="[%(asctime)s - %(levelname)s] %(message)s",
    level=_logging_level
)
logger = logging.getLogger(__name__)


def set_global_seed(seed):
    global _global_seed
    _global_seed = seed

    seed_everything(seed)


def get_global_seed():
    global _global_seed

    if _global_seed is None:
        logger.warning("Global seed is accessed but not initialized. Setting 42 as the global seed.")
        _global_seed = 42

    return _global_seed


def set_logging_level(level_name: str):
    level_name_lower = level_name.lower()

    if level_name_lower == "critical":
        level = logging.CRITICAL
    elif level_name_lower == "error":
        level = logging.ERROR
    elif level_name_lower == "warning":
        level = logging.WARNING
    elif level_name_lower == "info":
        level = logging.INFO
    elif level_name_lower == "debug":
        level = logging.DEBUG
    else:
        logger.warning(f"Unrecognized logging level name: {level_name} -- Setting logging level to INFO.")
        level = logging.INFO

    logger.setLevel(level)


def set_data_type(dtype_name):
    global _dtype

    if dtype_name in ["float", "float16", "torch.float16"]:
        _dtype = torch.float16
        logger.warning(f"torch.set_default_dtype does not officially support the given data type {dtype_name}."
                       f" Some operations may not work as expected.")
    elif dtype_name in ["bfloat", "bfloat16", "torch.bfloat16"]:
        _dtype = torch.bfloat16
        logger.warning(f"torch.set_default_dtype does not officially support the given data type {dtype_name}."
                       f" Some operations may not work as expected.")
    elif dtype_name in ["float", "float32", "torch.float32"]:
        _dtype = torch.float32
    elif dtype_name in ["double", "float64", "torch.float64"]:
        _dtype = torch.float64
    else:
        logger.warning(f"Unrecognized data type name: {dtype_name} -- Setting data type as torch.float32.")
        _dtype = torch.float32

    torch.set_default_dtype(_dtype)
