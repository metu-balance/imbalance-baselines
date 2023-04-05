import sys
import torch

from imbalance_baselines.config import Config
from imbalance_baselines.registry import Registry


device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")
cfg = Config(sys.argv[1])  # argv[1] should hold the path to config YAML
registry = Registry(cfg, static_transformations=True)

# Transformations are fully initialized from static parameters:
train_transform = registry.training_transform_module
test_transform = registry.testing_transform_module

train_dataset = registry.partial_modules['dataset'](train=True, transform=train_transform)
test_dataset = registry.partial_modules['dataset'](train=False, transform=test_transform)

train_dataloader = registry.partial_modules['dataloader'](dataset=train_dataset)
model = registry.partial_modules['model']()
optimizer = registry.partial_modules['optimizer'](params=model.parameters())
criterion = registry.partial_modules['loss']()

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
