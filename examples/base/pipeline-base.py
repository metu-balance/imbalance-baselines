# A sample pipeline describing a base training procedure with no imbalance mitigation methods.
# Note that some methods such as sampling-based ones could be described in the static .yaml configuration and be used
#   with this pipeline code.

import sys
import torch

from imbalance_baselines.config import Config
from imbalance_baselines.ConfigRegistry import Registry

# TODO: Use utils' parse config?

device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")
cfg = Config(sys.argv[1])  # argv[1] should hold the path to config YAML
registry = Registry(cfg, static_transofrmations=True)

# Transformations are fully initialized from static parameters:
train_transform = registry.training_transform_module
test_transform = registry.testing_transform_module

train_dataset = registry.partial_dataset_module(
    train=True, transform=train_transform)
test_dataset = registry.partial_dataset_module(
    train=False, transform=test_transform)

# TODO: Check dataloder config-registry format & usage...
train_dataloader = registry.partial_dataloader_module(dataset=train_dataset)
model = registry.partial_model_module()
optimizer = registry.partial_optimizer_module(params=model.parameters())
criterion = registry.partial_loss_module()  # Cross entropy does not require any parameters

# Intentionally simple training for test purposes only...
for epoch in range(5):
    print("EPOCH:", epoch)

    for i, (x, y) in enumerate(train_dataloader):
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        print("BATCH:", i)

# TODO: Extend with model save/load, evaluation, visualization (supporting wandb etc.)
