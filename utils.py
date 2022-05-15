import torch.nn as nn
import torch

def get_dataset_dict(dataset_name, model_name):
    if model_name == 'EEGNet' or model_name == 'BENDR':
        # for temporal features-based models, no downsampling
        if  dataset_name == 'AMIGOS':
            return {'temp_len':128, 'freq_len': 4, 'max_len': 14, 'name': 'AMIGOS'}
        if dataset_name == 'Search-Brainwave' or dataset_name == 'Example':
            return {'temp_len':1251, 'freq_len': 5, 'max_len': 62, 'name': 'Search-Brainwave'}
    if  dataset_name == 'AMIGOS':
        return {'temp_len':32, 'freq_len': 4, 'max_len': 14, 'name': 'AMIGOS'}
    if dataset_name == 'Search-Brainwave' or dataset_name == 'Example':
        return {'temp_len':62, 'freq_len': 5, 'max_len': 62, 'name': 'Search-Brainwave'}
    print("no dataset info")
    exit()

class MaskedMSELoss(nn.Module):
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
        self.mse_loss = nn.MSELoss(reduction=self.reduction)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
        masked_pred = torch.masked_select(y_pred, mask)
        masked_true = torch.masked_select(y_true, mask)
        return self.mse_loss(masked_pred, masked_true)


