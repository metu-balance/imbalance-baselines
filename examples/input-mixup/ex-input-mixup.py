import sys

import torch

from imbalance_baselines import datasets
from imbalance_baselines import set_global_seed, get_global_seed, set_logging_level, set_data_type
from imbalance_baselines.config import Config
from imbalance_baselines.models import ResNet32ManifoldMixup
from imbalance_baselines.loss_functions import MixupLoss
from imbalance_baselines.evaluation import evaluate


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
cfg = Config(sys.argv[1])

set_data_type(cfg.data_type)
set_global_seed(cfg.global_seed)
set_logging_level(cfg.logging_level)

EPOCH = cfg.Training.epoch_count
LR = cfg.Training.optimizer.params.lr
LR_WARMUP_EPOCHS = cfg.Training.optimizer.params.warmup_epochs
LR_DECAY_EPOCHS = cfg.Training.optimizer.params.lr_decay_epochs
LR_DECAY_RATE = cfg.Training.optimizer.params.lr_decay_rate
WEIGHT_DECAY_VALUE = cfg.Training.optimizer.params.weight_decay
MOMENTUM = cfg.Training.optimizer.params.momentum

ALPHA = cfg.Training.resnet32_mixup_params.beta_dist_alpha
FINETUNE_EPOCH = cfg.Training.resnet32_mixup_params.finetune_mixup_epochs


# Prepare dataset
train_dl, test_dl, dataset_info = datasets.generate_data(cfg)

model = ResNet32ManifoldMixup(num_layers=32, num_classes=dataset_info["class_count"], alpha=ALPHA,
                              seed=get_global_seed()).to(device)
criterion = torch.nn.CrossEntropyLoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=0, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY_VALUE)

loss_fn = MixupLoss(criterion, alpha=ALPHA)

model.train()
for j in range(EPOCH):
    total_loss = 0

    if j < LR_WARMUP_EPOCHS:
        for g in optimizer.param_groups:
            g['lr'] += LR / LR_WARMUP_EPOCHS

    if j + 1 == LR_DECAY_EPOCHS[0]:
        for g in optimizer.param_groups:
            g['lr'] *= LR_DECAY_RATE
    elif j + 1 == LR_DECAY_EPOCHS[1]:
        for g in optimizer.param_groups:
            g['lr'] *= LR_DECAY_RATE

    for i, (inp, target) in enumerate(train_dl):
        optimizer.zero_grad()

        inp = inp.to(device)
        target = target.to(device)

        loss = loss_fn(model(inp), target)  # Using with mix-up enabled
        loss.backward()
        optimizer.step()

        total_loss += loss.data.detach()

    print(f"Epoch: {j}, Input & Manifold Mix-up Loss: {total_loss / (i + 1)}, Num. of batches: {i}")

model.close_mixup()
loss_fn.close_mixup()

for j in range(FINETUNE_EPOCH):  # Fine-tuning after mix-up
    total_loss = 0

    for i, (inp, target) in enumerate(train_dl):
        optimizer.zero_grad()

        inp = inp.to(device)
        target = target.to(device)

        loss = loss_fn(model(inp), target)  # Using with mix-up disabled
        loss.backward()
        optimizer.step()

        total_loss += loss.data.detach()

    print(f"Epoch: {j}, Input & Manifold Mix-up Loss: {total_loss / (i + 1)}, Num. of batches: {i}")

# Convert model to library's format to use the standardized evaluation
train_results = [{
    "model": model,
    "loss_name": "ce_softmax",
    "model_name": "resnet32-manif-mu",
    "options": {}
}]

# Evaluate trained model's top-1 accuracy
model.eval()
evaluate(cfg=cfg, train_results=train_results, test_dl=test_dl, dataset_info=dataset_info, device=device)
