import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tmodels

from dnn.models.builder import LOSSES
from dnn.models.builder import build_losses
from dnn.models.losses.pixelwise_loss import l1_loss


@LOSSES.register_module()
class CosineColorLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(CosineColorLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, x1, x2):
        norm_x1 = F.normalize(x1, dim=1)
        norm_x2 = F.normalize(x2, dim=1)
        cosine_map = torch.sum(norm_x1 * norm_x2, dim=1)
        return self.loss_weight * (1 - cosine_map.mean())


@LOSSES.register_module()
class GradLoss(nn.Module):
    """Gradient loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super().__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        b, c, h, w = target.size()
        kx = torch.Tensor([[1, 0, -1],
                           [2, 0, -2],
                           [1, 0, -1]]).view(1, 1, 3, 3).to(target)
        ky = torch.Tensor([[1, 2, 1],
                           [0, 0, 0],
                           [-1, -2, -1]]).view(1, 1, 3, 3).to(target)

        pred_grad_x = F.conv2d(pred.view(b*c, 1, h, w), kx, padding=1).view(b, c, h, w)
        pred_grad_y = F.conv2d(pred.view(b*c, 1, h, w), ky, padding=1).view(b, c, h, w)
        target_grad_x = F.conv2d(target.view(b*c, 1, h, w), kx, padding=1).view(b, c, h, w)
        target_grad_y = F.conv2d(target.view(b*c, 1, h, w), ky, padding=1).view(b, c, h, w)

        loss = (
            l1_loss(
                pred_grad_x, target_grad_x, weight, reduction=self.reduction) +
            l1_loss(
                pred_grad_y, target_grad_y, weight, reduction=self.reduction))
        return loss * self.loss_weight


@LOSSES.register_module()
class TanhL1Loss(nn.Module):

    def __init__(self, loss_weight):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, x, y, weight=None):
        loss = l1_loss(torch.tanh(x), torch.tanh(y), weight)
        return loss * self.loss_weight
