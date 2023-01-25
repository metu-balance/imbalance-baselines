import sys

from imbalance_baselines import evaluation
from imbalance_baselines import datasets
from imbalance_baselines import training
from imbalance_baselines import set_global_seed
from imbalance_baselines.config import Config

import torch.cuda

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
cfg = Config(sys.argv[1])  # argv[1] should hold the path to config YAML

set_global_seed(cfg.global_seed)

train_dl, test_dl, dataset_info = datasets.generate_data(cfg)
# dataset_info holds: class_count, train_class_sizes, test_class_sizes
print(f"Formed dataset and dataloaders with {dataset_info['class_count']} classes.")

training_results = training.train_models(cfg, train_dl, dataset_info, device=device)
print("Training completed, evaluating models...")

evaluation.evaluate(cfg, training_results, test_dl, dataset_info, device=device)
