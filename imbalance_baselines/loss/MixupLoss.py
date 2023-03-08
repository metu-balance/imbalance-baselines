import numpy as np
import torch


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
