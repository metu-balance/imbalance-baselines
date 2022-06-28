import numpy as np
import torch

from torch.utils.data import DataLoader


# TODO [3]: This weight function should be renamed/refactored to better represent the
#   specific method it uses (effective no. of samples?)
def get_weights(class_sizes, beta=0, dtype: torch.dtype = torch.double,
                device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """Get normalized weight (inverse of effective number of samples) per class."""
    
    class_sizes = torch.as_tensor(
        class_sizes,
        #dtype=torch.long,
        device=device
    )

    class_cnt = class_sizes.shape[0]
    
    weights = torch.as_tensor(
        [1 - beta] * class_cnt,
        dtype=dtype,
        device=device
    )
    
    weights = torch.div(
        weights, 1 - torch.pow(beta, class_sizes)
    ).to(device)
    
    # Normalize the weights
    weights = torch.mul(weights, class_cnt / torch.sum(weights)).to(device)
    
    weights.requires_grad = False
    
    return weights.double()


def get_size_per_class(data, num_classes=10):
    size = torch.tensor([0] * num_classes, dtype=torch.float32)

    for feature, label in data:
        size[label] += 1

    return size


# TODO: This should be renamed as a specific evaluation method. The overall get_accuracy function should
#   iterate over trained model & eval. method preference pairs.
def get_accuracy(test_data: DataLoader, model, class_sizes:[int],
                 device: torch.device = torch.device("cpu"), calc_avg=True, calc_perclass=True, top=1):
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
        
        for num_batch, (inp, target) in enumerate(test_data):
            inp = inp.double().to(device)
            target = target.to(device)
            output = model(inp).to(device)
            
            if top == 1:
                result = (torch.argmax(output, dim=1) == target)
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
        
        # TODO: avg_acc is initialized as a float but is assigned a tensor throughout evaluation.
        #   Change avg_acc type to tensor & return contained value, or process as a float from the beginning.
        return avg_acc, per_class_acc.tolist()
