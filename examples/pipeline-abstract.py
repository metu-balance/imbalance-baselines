import sys

from imbalance_baselines.config import Config
from imbalance_baselines.ConfigRegistry import Registry


def class_balance_cal(dataset):
    class_weights = [31, 31, 31, 31]
    return class_weights

path = "./config-abstract.yaml"
cfg = Config(sys.argv[1])  # argv[1] should hold the path to config YAML
registar = Registry(cfg)

registar.read_config()

partial_dataset = registar.partial_dataset_module
partial_dataloader = registar.partial_dataloader_module
partial_optimizer = registar.partial_optimizer_module
partial_model = registar.partial_model_module
partial_loss = registar.partial_loss_module

dataset = partial_dataset()
dataloader = partial_dataloader(dataset = dataset)
model = partial_model()
optimizer = partial_optimizer(params = model.parameters())
criterion = partial_loss(weights = class_balance_cal(dataset))

for epoch in range(5):
    for (x, y) in enumerate(dataloader):
        
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        
