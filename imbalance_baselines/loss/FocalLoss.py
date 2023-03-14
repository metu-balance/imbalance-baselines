import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.ops import sigmoid_focal_loss


class FocalLoss:
    def __init__(self, gamma=0, device: torch.device = torch.device("cpu"), custom_implementation=False):
        self.gamma = gamma
        self.custom_implementation = custom_implementation

        self.device = device

    def __call__(self, z, lbl, alpha=None, gamma=None, reduction="sum"):
        """Return the focal loss tensor of shape [BATCH_SIZE] for given model & lbl.s.

            Args:
              z: Predictions tensor of shape [BATCH_SIZE, label_count], output of ResNet
              lbl: Labels tensor of shape [BATCH_SIZE]
              alpha: Class balance cb_weights tensor of shape [lable_count]. Taken 1 for all classes
                if None is given.
              gamma: Focal loss parameter (if 0, loss is equivalent to sigmoid ce. loss)
            """
        if self.custom_implementation:
            # Not BATCH_SIZE: The last batch might be smaller
            batch_size = z.shape[0]
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

            if gamma is None:
                gamma = self.gamma

            if gamma == 0:
                modulator = 1
            else:
                modulator = torch.exp(
                    -gamma * torch.mul(lbl, z).to(self.device) - gamma *
                    torch.log1p(torch.exp(-1.0 * z)).to(self.device)
                )

            # Sum the value of each class in each batch. The shape is reduced from
            #  [BATCH_SIZE, label_count] to [BATCH_SIZE].
            unweighted_focal_loss = -torch.sum(torch.mul(modulator, cross_entpy), 1).to(
                self.device
            )
            weighted_focal_loss = torch.mul(
                alpha, unweighted_focal_loss).to(self.device)

            # Normalize by the positive sample count:
            weighted_focal_loss /= torch.sum(lbl)

            if reduction == "sum":
                return torch.sum(weighted_focal_loss)
            elif reduction == "mean":
                return torch.mean(weighted_focal_loss)
            elif reduction == "none":
                return weighted_focal_loss
            else:
                raise ValueError(
                    f"Unrecognized reduction type: {reduction}. Should be one of 'sum', 'mean', 'none'.")
        else:
            # FIXME: I/O shapes are inconsistent. Pass through torch.sum before returning?
            return sigmoid_focal_loss(inputs=z, targets=lbl, alpha=alpha, gamma=gamma, reduction=reduction)
