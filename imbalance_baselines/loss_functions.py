import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import utils
from .datasets import get_cb_weights
from torchvision.ops import sigmoid_focal_loss


class CostSensitiveCrossEntropy:
    def __init__(self, dataset, num_classes: int, beta: float, device: torch.device = torch.device("cpu")):
        class_sizes = utils.get_size_per_class(dataset, num_classes)
        weights = class_sizes.min() / class_sizes
        self.CS_CE = nn.CrossEntropyLoss(weight=weights.to(device), reduction='Sum')
        self.weights = weights
        self.beta = beta
    
    def __call__(self, logits, labels):
        batch_size = labels.shape[0]
        loss = self.CS_CE(logits, labels) / batch_size
        
        return loss


class ClassBalancedCrossEntropy:
    def __init__(self, dataset, num_classes: int, beta: float):
        class_sizes = utils.get_size_per_class(dataset, num_classes)
        weights = get_cb_weights(class_sizes, beta)
        self.CB_CE = torch.nn.CrossEntropyLoss(weight=weights, reduction='Sum')
        self.weights = weights
        self.beta = beta
    
    def __call__(self, logits, labels):
        batch_size = labels.shape[0]
        loss = self.CB_CE(logits, labels) / batch_size
        
        return loss


class InputMixup:
    def __init__(self, alpha, model, criterion):
        self.model = model
        self.criterion = criterion
        self.alpha = alpha
    
    def __call__(self, image, label):
        l = np.random.beta(self.alpha, self.alpha)
        idx = torch.randperm(image.size(0))
        image_a, image_b = image, image[idx]
        label_a, label_b = label, label[idx]
        mixed_image = l * image_a + (1 - l) * image_b
        
        output = self.model(mixed_image)
        loss = l * self.criterion(output, label_a) + (1 - l) * self.criterion(output, label_b)
        
        return loss


class FocalLoss:
    def __init__(self, device: torch.device = torch.device("cpu"), custom_implementation=False):
        self.custom_implementation = custom_implementation

        if not custom_implementation:
            self.device = device
    
    def __call__(self, z, lbl, alpha=None, gamma=0, reduction="sum"):
        """Return the focal loss tensor of shape [BATCH_SIZE] for given model & lbl.s.

            Args:
              z: Predictions tensor of shape [BATCH_SIZE, label_count], output of ResNet
              lbl: Labels tensor of shape [BATCH_SIZE]
              alpha: Class balance cb_weights tensor of shape [lable_count]. Taken 1 for all classes
                if None is given.
              gamma: Focal loss parameter (if 0, loss is equivalent to sigmoid ce. loss)
            """
        if self.custom_implementation:
            if reduction != "sum":
                raise ValueError("Currently, only sum reduction is implemented in the custom focal loss.")  # TODO

            batch_size = z.shape[0]  # Not BATCH_SIZE: The last batch might be smaller
            lbl_cnt = z.shape[1]

            # "Decode" labels tensor to make its shape [BATCH_SIZE, label_count]:
            lbl = F.one_hot(lbl, num_classes=lbl_cnt)

            if alpha is None:
                alpha = torch.as_tensor([1] * batch_size, device=self.device)
            else:  # Get cb_weights for each image in batch
                alpha = (alpha * lbl).sum(axis=1)

            lbl_bool = lbl.type(torch.bool)  # Cast to bool for torch.where()
            z_t = torch.where(lbl_bool, z, -z).to(self.device)

            logsig = nn.LogSigmoid()

            cross_entpy = logsig(z_t).to(self.device)

            if gamma:
                modulator = torch.exp(
                    -gamma * torch.mul(lbl, z).to(self.device) - gamma * torch.log1p(torch.exp(-1.0 * z)).to(self.device)
                )
            else:
                modulator = 1

            # Sum the value of each class in each batch. The shape is reduced from
            #  [BATCH_SIZE, label_count] to [BATCH_SIZE].
            unweighted_focal_loss = -torch.sum(torch.mul(modulator, cross_entpy), 1).to(
                self.device
            )
            weighted_focal_loss = torch.mul(alpha, unweighted_focal_loss).to(self.device)

            # Normalize by the positive sample count:
            weighted_focal_loss /= torch.sum(lbl)

            return torch.sum(weighted_focal_loss)  # TODO: Implement other reduction methods as well
        else:
            return sigmoid_focal_loss(inputs=z, targets=lbl, alpha=alpha, gamma=gamma, reduction=reduction)


class MixupLoss:
    def __init__(self, criterion, alpha=1, seed=12649):
        self.alpha = alpha
        self.criterion = criterion
        self.random_gen = np.random.default_rng(seed)
        self.mixup = True
    
    def __call__(self, logits, labels):
        if self.mixup:
            lamb = self.random_gen.beta(self.alpha, self.alpha)
            idx = torch.randperm(labels.size(0))
            
            label_a, label_b = labels, labels[idx]
            loss = lamb * self.criterion(logits, label_a) + (1 - lamb) * self.criterion(logits, label_b)
        
        else:
            loss = self.criterion(logits, labels)
        
        return loss

    def close_mixup(self):
        self.mixup = False
