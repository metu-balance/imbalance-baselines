import torch
import torch.nn as nn

from .. import utils


class CostSensitiveCrossEntropy:
    def __init__(self, dataset, num_classes: int, beta: float, device: torch.device = torch.device("cpu")):
        class_sizes = utils.get_size_per_class(dataset, num_classes)
        weights = class_sizes.min() / class_sizes
        self.CS_CE = nn.CrossEntropyLoss(
            weight=weights.to(device), reduction='Sum')
        self.weights = weights
        self.beta = beta

    def __call__(self, logits, labels):
        batch_size = labels.shape[0]
        loss = self.CS_CE(logits, labels) / batch_size

        return loss
