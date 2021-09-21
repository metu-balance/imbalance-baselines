import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader


# TODO: Deprecate if unneeded (see: get_weights in ClassBalancedCrossEntropy)
#   (Or, just remove that method version as it is already static)
def get_weights(class_sizes, beta=0, device: torch.device = torch.device("cpu")):
    """Get normalized weight per class."""
    
    class_cnt = len(class_sizes)
    class_sizes = torch.as_tensor(
        class_sizes,
        dtype=torch.float32,
        device=device
    )
    
    weights = torch.as_tensor(
        [1 - beta] * class_cnt,
        dtype=torch.float32,
        device=device
    )
    
    weights = torch.div(
        weights, 1 - torch.pow(beta, class_sizes)
    ).to(device)
    
    # Normalize the weights
    weights = torch.mul(weights, class_cnt / torch.sum(weights)).to(device)
    
    return weights.double()


def get_accuracy(test_data:DataLoader, model, class_sizes:[int],
                 device:torch.device, calc_avg=True, calc_perclass=True, top=1):
    """Return a tuple containing the accuracy values of a given model.
  
    The first element of the returned tuple is the average accuracy of the model.
    The second element is a tensor containing the accuracy of each class
      separately.
    
    The parameters calc_avg and calc_perclass exist for performance adjustments.
    If calc_avg is False, the average accuracy isn't calculated and left as 0.
    If calc_perclass is False, the per-class accuracies aren't calculated and left
      as 0.
    """
    
    total_size = 0
    
    with torch.no_grad():
        num_labels = len(class_sizes)
        
        per_class_acc = torch.zeros(num_labels, dtype=torch.float32, device=device)
        avg_acc = float(0)
        
        for num_batch, (input, target) in enumerate(test_data):
            input = input.double().to(device)
            target = target.to(device)
            output = model(input).to(device)
            
            if top == 1:
                result = (torch.argmax(output, dim = 1) == target)
            else:
                result = []
                top_preds = np.argpartition(output, -top)[:, -top:]
                
                for i, t in enumerate(target):
                    result.append(t.item() in top_preds[i])
                
                result = torch.Tensor(result)
            
            batch_len = result.shape[0]
            
            if calc_avg:
                avg_acc += result.sum()
            
            if calc_perclass:
                for i in range(batch_len):
                    per_class_acc[target[i]] += result[i]
            
            total_size += batch_len
        
        # Average accuracy of the whole test dataset
        if calc_avg: avg_acc /= total_size
        
        if calc_perclass:
            for i in range(num_labels):
                # Average accuracy of every class separately
                per_class_acc[i] /= class_sizes[i]
        
        return (avg_acc.item(), per_class_acc.tolist())

