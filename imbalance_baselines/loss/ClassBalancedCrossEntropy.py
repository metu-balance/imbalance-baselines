import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import utils
from .datasets import get_cb_weights
from torchvision.ops import sigmoid_focal_loss


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