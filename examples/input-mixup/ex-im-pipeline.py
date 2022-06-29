import sys
import torch.cuda

from imbalance_baselines import evaluation
from imbalance_baselines import datasets
from imbalance_baselines import training
from imbalance_baselines import utils
from imbalance_baselines.config import Config

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# TODO: Use argparse
cfg = Config(sys.argv[1])  # argv[1] should hold the path to config YAML


