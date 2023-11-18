import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-05):
        super().__init__()
        self.smooth = smooth

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor):
        reduce_axis = torch.arange(2, len(y_true.shape)).tolist()
        smooth = 1e-05
        intersection = torch.sum((y_true * y_pred), dim=reduce_axis)
        coeff = (2. * intersection + smooth) / (
                torch.sum(y_true, dim=reduce_axis) + torch.sum(y_pred, dim=reduce_axis) + smooth)
        return 1. - torch.mean(coeff)
