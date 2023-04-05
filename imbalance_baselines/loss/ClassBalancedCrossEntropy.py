import torch


def get_cb_weights(class_sizes, beta=0) -> torch.Tensor:
    """Get normalized weight (inverse of effective number of samples) per class."""

    class_sizes = torch.as_tensor(
        class_sizes,
        # dtype=torch.long,
    )

    class_cnt = class_sizes.shape[0]

    weights = torch.as_tensor(
        [1 - beta] * class_cnt
    )

    weights = torch.div(
        weights, 1 - torch.pow(beta, class_sizes)
    )

    # Normalize the cb_weights
    weights = torch.mul(weights, class_cnt / torch.sum(weights))

    weights.requires_grad = False

    return weights


def get_size_per_class(data, num_classes=10):
    size = torch.tensor([0] * num_classes, dtype=torch.float32)

    for feature, label in data:
        size[label] += 1

    return size


class ClassBalancedCrossEntropy:
    def __init__(self, dataset, num_classes: int, beta: float):
        class_sizes = get_size_per_class(dataset, num_classes)
        weights = get_cb_weights(class_sizes, beta)
        self.CB_CE = torch.nn.CrossEntropyLoss(weight=weights, reduction='sum')
        self.weights = weights
        self.beta = beta

    def __call__(self, logits, labels):
        batch_size = labels.shape[0]
        loss = self.CB_CE(logits, labels) / batch_size

        return loss
