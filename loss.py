import torch
import torch.nn as nn
from torch.nn import functional as F


def get_loss_module(task_name):
    if (task_name == "imputation") or (task_name == "transduction"):
        return MaskedMSELoss(reduction='none')  # outputs loss for each batch element
    if task_name == "regression":
        return nn.MSELoss(reduction='none')  # outputs loss for each batch sample
    else:
        raise ValueError("Loss module for task '{}' does not exist".format(task_name))


def l2_reg_loss(model):
    """Returns the squared L2 norm of output layer of given model"""
    for name, param in model.named_parameters():
        if name == 'output_layer.weight':
            return torch.sum(torch.square(param))


class MaskedMSELoss(nn.Module):
    """ Masked MSE Loss
    """
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
        self.mse_loss = nn.MSELoss(reduction=self.reduction)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
        masked_pred = torch.masked_select(y_pred, mask)
        masked_true = torch.masked_select(y_true, mask)
        return self.mse_loss(masked_pred, masked_true)
