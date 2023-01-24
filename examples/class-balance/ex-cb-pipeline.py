import sys

from imbalance_baselines import evaluation
from imbalance_baselines import datasets
from imbalance_baselines import training
from imbalance_baselines import utils
from imbalance_baselines.config import Config

import torch.cuda

beta = 0.9999  # TODO: Pass beta thru. config.
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

cfg = Config(sys.argv[1])  # argv[1] should hold the path to config YAML

train_dl, test_dl, class_cnt, train_class_sizes, test_class_sizes = datasets.generate_data(cfg)

weights = utils.get_weights(train_class_sizes, beta, device=device)
weights.requires_grad = False

print("Got weights:", weights)

training_results = training.train_models(cfg, train_dl, class_cnt, weights, device=device)
evaluation.evaluate(cfg, training_results, test_dl, test_class_sizes, device=device)

print("Done!")
