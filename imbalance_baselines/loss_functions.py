import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CostSensitiveCrossEntropy:
    def __init__(self, dataset, num_classes: int, beta: float, device='cpu'):
        sizes = self.get_size_per_class(dataset, num_classes)
        weights = sizes.min() / sizes
        self.CS_CE = nn.CrossEntropyLoss(weight=weights.to(device), reduction='Sum')
        self.weights = weights
        self.beta = beta
    
    def __call__(self, logits, labels):
        batch_size = labels.shape[0]
        loss = self.CS_CE(logits, labels) / batch_size
        
        return loss
    
    @staticmethod
    def get_size_per_class(data, num_classes=10):
        size = torch.tensor([0] * num_classes, dtype=torch.float32)
        
        for feature, label in data:
            size[label] += 1
        
        return size


class ClassBalancedCrossEntropy:
    def __init__(self, dataset, num_classes: int, beta: float):
        sizes = self.get_size_per_class(dataset, num_classes)
        weights = self.get_weights(sizes, beta)
        self.CB_CE = torch.nn.CrossEntropyLoss(weight=weights, reduction='Sum')
        self.weights = weights
        self.beta = beta
    
    def __call__(self, logits, labels):
        batch_size = labels.shape[0]
        loss = self.CB_CE(logits, labels) / batch_size
        
        return loss
    
    @staticmethod
    def get_weights(size, beta=0.):
        num_classes = size.shape[0]
        numerator = torch.tensor(1 - beta, dtype=torch.float32, requires_grad=False)
        denominator = 1 - beta ** size
        
        weights = numerator / denominator
        weights *= num_classes / weights.sum()
        weights.requires_grad = False
        
        return weights
    
    @staticmethod
    def get_size_per_class(data, num_classes=10):
        size = torch.tensor([0] * num_classes, dtype=torch.float32)
        
        for feature, label in data:
            size[label] += 1
        
        return size


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


# TODO: Convert to class?
def focal_loss(z, lbl, alpha=None, gamma=0, device: torch.device = torch.device("cpu")):
    """Return the focal loss tensor of shape [BATCH_SIZE] for given model & lbl.s.
  
    Args:
      z: Predictions tensor of shape [BATCH_SIZE, label_count], output of ResNet
      lbl: Labels tensor of shape [BATCH_SIZE]
      alpha: Class balance weights tensor of shape [lable_count]
      gamma: Focal loss parameter (if 0, loss is equivalent to sigmoid ce. loss)
      device: A torch.device object denoting the device to operate on
    """
    
    z = z.double()
    
    batch_size = z.shape[0]  # Not BATCH_SIZE: The last batch might be smaller
    lbl_cnt = z.shape[1]
    
    # "Decode" labels tensor to make its shape [BATCH_SIZE, label_count]:
    lbl = F.one_hot(lbl, num_classes=lbl_cnt)
    
    if alpha is None:
        alpha = torch.as_tensor([1] * batch_size, device=device)
    else:
        # Get weights for each image in batch
        
        alpha = (alpha * lbl).sum(axis=1)
        
    lbl_bool = lbl.type(torch.bool)  # Cast to bool for torch.where()
    z_t = torch.where(lbl_bool, z, -z).to(device)
    
    sig = nn.Sigmoid()
    logsig = nn.LogSigmoid()
    
    cross_entpy = logsig(z_t).to(device)
    
    if gamma:
        modulator = torch.exp(
            -gamma * torch.mul(lbl, z).to(device) - gamma * torch.log1p(torch.exp(-1.0 * z)).to(device)
        )
    else:
        modulator = 1
    
    # Sum the value of each class in each batch. The shape is reduced from
    #  [BATCH_SIZE, label_count] to [BATCH_SIZE].
    unweighted_focal_loss = -torch.sum(torch.mul(modulator, cross_entpy), 1).to(
        device
    )
    weighted_focal_loss = torch.mul(alpha, unweighted_focal_loss).to(device)
    
    # Normalize by the positive sample count:
    weighted_focal_loss /= torch.sum(lbl)
    
    return torch.sum(weighted_focal_loss)
