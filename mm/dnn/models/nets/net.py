import torch
import torch.nn as nn
import torch.nn.functional as F
from ..builder import NETS


@NETS.register_module()
class ResBlock(nn.Module):
    """ResBlock for Hourglass.

    It has a style of:

    ::

        ---Conv-ReLU-Conv-Conv-+-
         |_________Conv________|

        or

        ---Conv-ReLU-Conv-Conv-+-
         |_____________________|

    Args:
        in_channels (int): Number of channels in the input features.
        out_channels (int): Number of channels in the output features.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels // 2, out_channels // 2, 3, stride=1, padding=1),
            nn.Conv2d(out_channels // 2, out_channels, 1))
        if in_channels == out_channels:
            self.skip_layer = None
        else:
            self.skip_layer = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """

        residual = self.conv_block(x)
        if self.skip_layer:
            x = self.skip_layer(x)
        return x + residual


@NETS.register_module()
class Hourglass(nn.Module):
    """Hourglass model for face landmark.

    It is a recursive model.

    Args:
        depth (int): Depth of Hourglass, the number of recursions.
        mid_channels (int): Number of channels in the intermediate features.
    """

    def __init__(self, depth, mid_channels):
        super().__init__()
        self.up1 = ResBlock(mid_channels, mid_channels)
        self.pool = nn.MaxPool2d(2, 2)
        self.low1 = ResBlock(mid_channels, mid_channels)
        if depth == 1:
            self.low2 = ResBlock(mid_channels, mid_channels)
        else:
            self.low2 = Hourglass(depth - 1, mid_channels)
        self.low3 = ResBlock(mid_channels, mid_channels)

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """

        up1 = self.up1(x)
        low1 = self.low1(self.pool(x))
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2 = nn.functional.interpolate(
            low3, scale_factor=2, mode='bilinear', align_corners=True)
        return up1 + up2


@NETS.register_module()
class Encoding(nn.Module):
    """Encoding Layer: a learnable residual encoder.

    Input is of shape  (batch_size, channels, height, width).
    Output is of shape (batch_size, num_codes, channels).

    Args:
        channels: dimension of the features or feature channels
        num_codes: number of code words
    """

    def __init__(self, channels, num_codes):
        super(Encoding, self).__init__()
        # init codewords and smoothing factor
        self.channels, self.num_codes = channels, num_codes
        std = 1. / ((num_codes * channels)**0.5)
        # [num_codes, channels]
        self.codewords = nn.Parameter(
            torch.empty(num_codes, channels,
                        dtype=torch.float).uniform_(-std, std),
            requires_grad=True)
        # [num_codes]
        self.scale = nn.Parameter(
            torch.empty(num_codes, dtype=torch.float).uniform_(-1, 0),
            requires_grad=True)

    @staticmethod
    def scaled_l2(x, codewords, scale):
        num_codes, channels = codewords.size()
        batch_size = x.size(0)
        reshaped_scale = scale.view((1, 1, num_codes))
        expanded_x = x.unsqueeze(2).expand(
            (batch_size, x.size(1), num_codes, channels))
        reshaped_codewords = codewords.view((1, 1, num_codes, channels))

        scaled_l2_norm = reshaped_scale * (
            expanded_x - reshaped_codewords).pow(2).sum(dim=3)
        return scaled_l2_norm

    @staticmethod
    def aggregate(assignment_weights, x, codewords):
        num_codes, channels = codewords.size()
        reshaped_codewords = codewords.view((1, 1, num_codes, channels))
        batch_size = x.size(0)

        expanded_x = x.unsqueeze(2).expand(
            (batch_size, x.size(1), num_codes, channels))
        encoded_feat = (assignment_weights.unsqueeze(3) *
                        (expanded_x - reshaped_codewords)).sum(dim=1)
        return encoded_feat

    def forward(self, x):
        assert x.dim() == 4 and x.size(1) == self.channels
        # [batch_size, channels, height, width]
        batch_size = x.size(0)
        # [batch_size, height x width, channels]
        x = x.view(batch_size, self.channels, -1).transpose(1, 2).contiguous()
        # assignment_weights: [batch_size, channels, num_codes]
        assignment_weights = F.softmax(
            self.scaled_l2(x, self.codewords, self.scale), dim=2)
        # aggregate
        encoded_feat = self.aggregate(assignment_weights, x, self.codewords)
        return encoded_feat

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(Nx{self.channels}xHxW =>Nx{self.num_codes}' \
                    f'x{self.channels})'
        return repr_str