import torch


def parse_cfg_str(inp, casttype):
    return casttype(inp) if isinstance(inp, str) else inp


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
    weights = torch.mul(weights, class_cnt / torch.sum(weights))
    
    weights.requires_grad = False
    
    # TODO: Check whether data type of weights are dtype or not
    return weights.to(device)


def get_size_per_class(data, num_classes=10):
    size = torch.tensor([0] * num_classes, dtype=torch.float32)

    for feature, label in data:
        size[label] += 1

    return size
