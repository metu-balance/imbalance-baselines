import sys
import torch

from imbalance_baselines.config import Config
from imbalance_baselines.ConfigRegistry import Registry

# TODO: Use utils' parse config?

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
cfg = Config(sys.argv[1])  # argv[1] should hold the path to config YAML
registry = Registry(cfg)

partial_dataset = registry.partial_dataset_module
partial_dataloader = registry.partial_dataloader_module
partial_optimizer = registry.partial_optimizer_module
partial_model = registry.partial_model_module
partial_loss = registry.partial_loss_module

full_train_transform = registry.full_training_transform_module
full_test_transform = registry.full_testing_transform_module

train_dataset = partial_dataset(train=True, transform=full_train_transform)
test_dataset = partial_dataset(train=False, transform=full_test_transform)
train_dataloader = partial_dataloader(dataset=train_dataset)
model = partial_model()
optimizer = partial_optimizer(params=model.parameters())
criterion = partial_loss(device=device)

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
        # exit()
