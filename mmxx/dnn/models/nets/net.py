import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from ..builder import NETS


@NETS.register_module()
class UNetDiscriminatorWithSpectralNorm(nn.Module):
    """A U-Net discriminator with spectral normalization.
    Args:
        in_channels (int): Channel number of the input.
        mid_channels (int, optional): Channel number of the intermediate
            features. Default: 64.
        skip_connection (bool, optional): Whether to use skip connection.
            Default: True.
    """

    def __init__(self, in_channels, mid_channels=64, skip_connection=True):

        super().__init__()

        self.skip_connection = skip_connection

        self.conv_0 = nn.Conv2d(
            in_channels, mid_channels, kernel_size=3, stride=1, padding=1)

        # downsample
        self.conv_1 = spectral_norm(
            nn.Conv2d(mid_channels, mid_channels * 2, 4, 2, 1, bias=False))
        self.conv_2 = spectral_norm(
            nn.Conv2d(mid_channels * 2, mid_channels * 4, 4, 2, 1, bias=False))
        self.conv_3 = spectral_norm(
            nn.Conv2d(mid_channels * 4, mid_channels * 8, 4, 2, 1, bias=False))

        # upsample
        self.conv_4 = spectral_norm(
            nn.Conv2d(mid_channels * 8, mid_channels * 4, 3, 1, 1, bias=False))
        self.conv_5 = spectral_norm(
            nn.Conv2d(mid_channels * 4, mid_channels * 2, 3, 1, 1, bias=False))
        self.conv_6 = spectral_norm(
            nn.Conv2d(mid_channels * 2, mid_channels, 3, 1, 1, bias=False))

        # final layers
        self.conv_7 = spectral_norm(
            nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=False))
        self.conv_8 = spectral_norm(
            nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=False))
        self.conv_9 = nn.Conv2d(mid_channels, 1, 3, 1, 1)

        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, img):
        """Forward function.
        Args:
            img (Tensor): Input tensor with shape (n, c, h, w).
        Returns:
            Tensor: Forward results.
        """

        feat_0 = self.lrelu(self.conv_0(img))

        # downsample
        feat_1 = self.lrelu(self.conv_1(feat_0))
        feat_2 = self.lrelu(self.conv_2(feat_1))
        feat_3 = self.lrelu(self.conv_3(feat_2))

        # upsample
        feat_3 = self.upsample(feat_3)
        feat_4 = self.lrelu(self.conv_4(feat_3))
        if self.skip_connection:
            feat_4 = feat_4 + feat_2

        feat_4 = self.upsample(feat_4)
        feat_5 = self.lrelu(self.conv_5(feat_4))
        if self.skip_connection:
            feat_5 = feat_5 + feat_1

        feat_5 = self.upsample(feat_5)
        feat_6 = self.lrelu(self.conv_6(feat_5))
        if self.skip_connection:
            feat_6 = feat_6 + feat_0

        # final layers
        out = self.lrelu(self.conv_7(feat_6))
        out = self.lrelu(self.conv_8(out))

        return self.conv_9(out)

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


class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


@NETS.register_module()
class VGGStyleDiscriminator(nn.Module):
    """VGG style discriminator with input size 128 x 128 or 256 x 256.

    It is used to train SRGAN, ESRGAN, and VideoGAN.

    Args:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_feat (int): Channel number of base intermediate features.Default: 64.
    """

    def __init__(self, num_in_ch, num_feat, input_size=128):
        super(VGGStyleDiscriminator, self).__init__()
        self.input_size = input_size
        assert self.input_size == 128 or self.input_size == 256, (
            f'input size must be 128 or 256, but received {input_size}')

        self.conv0_0 = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1, bias=True)
        self.conv0_1 = nn.Conv2d(num_feat, num_feat, 4, 2, 1, bias=False)
        self.bn0_1 = nn.BatchNorm2d(num_feat, affine=True)

        self.conv1_0 = nn.Conv2d(num_feat, num_feat * 2, 3, 1, 1, bias=False)
        self.bn1_0 = nn.BatchNorm2d(num_feat * 2, affine=True)
        self.conv1_1 = nn.Conv2d(num_feat * 2, num_feat * 2, 4, 2, 1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(num_feat * 2, affine=True)

        self.conv2_0 = nn.Conv2d(num_feat * 2, num_feat * 4, 3, 1, 1, bias=False)
        self.bn2_0 = nn.BatchNorm2d(num_feat * 4, affine=True)
        self.conv2_1 = nn.Conv2d(num_feat * 4, num_feat * 4, 4, 2, 1, bias=False)
        self.bn2_1 = nn.BatchNorm2d(num_feat * 4, affine=True)

        self.conv3_0 = nn.Conv2d(num_feat * 4, num_feat * 8, 3, 1, 1, bias=False)
        self.bn3_0 = nn.BatchNorm2d(num_feat * 8, affine=True)
        self.conv3_1 = nn.Conv2d(num_feat * 8, num_feat * 8, 4, 2, 1, bias=False)
        self.bn3_1 = nn.BatchNorm2d(num_feat * 8, affine=True)

        self.conv4_0 = nn.Conv2d(num_feat * 8, num_feat * 8, 3, 1, 1, bias=False)
        self.bn4_0 = nn.BatchNorm2d(num_feat * 8, affine=True)
        self.conv4_1 = nn.Conv2d(num_feat * 8, num_feat * 8, 4, 2, 1, bias=False)
        self.bn4_1 = nn.BatchNorm2d(num_feat * 8, affine=True)

        if self.input_size == 256:
            self.conv5_0 = nn.Conv2d(num_feat * 8, num_feat * 8, 3, 1, 1, bias=False)
            self.bn5_0 = nn.BatchNorm2d(num_feat * 8, affine=True)
            self.conv5_1 = nn.Conv2d(num_feat * 8, num_feat * 8, 4, 2, 1, bias=False)
            self.bn5_1 = nn.BatchNorm2d(num_feat * 8, affine=True)

        self.linear1 = nn.Linear(num_feat * 8 * 4 * 4, 100)
        self.linear2 = nn.Linear(100, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        assert x.size(2) == self.input_size, (f'Input size must be identical to input_size, but received {x.size()}.')

        feat = self.lrelu(self.conv0_0(x))
        feat = self.lrelu(self.bn0_1(self.conv0_1(feat)))  # output spatial size: /2

        feat = self.lrelu(self.bn1_0(self.conv1_0(feat)))
        feat = self.lrelu(self.bn1_1(self.conv1_1(feat)))  # output spatial size: /4

        feat = self.lrelu(self.bn2_0(self.conv2_0(feat)))
        feat = self.lrelu(self.bn2_1(self.conv2_1(feat)))  # output spatial size: /8

        feat = self.lrelu(self.bn3_0(self.conv3_0(feat)))
        feat = self.lrelu(self.bn3_1(self.conv3_1(feat)))  # output spatial size: /16

        feat = self.lrelu(self.bn4_0(self.conv4_0(feat)))
        feat = self.lrelu(self.bn4_1(self.conv4_1(feat)))  # output spatial size: /32

        if self.input_size == 256:
            feat = self.lrelu(self.bn5_0(self.conv5_0(feat)))
            feat = self.lrelu(self.bn5_1(self.conv5_1(feat)))  # output spatial size: / 64

        # spatial size: (4, 4)
        feat = feat.view(feat.size(0), -1)
        feat = self.lrelu(self.linear1(feat))
        out = self.linear2(feat)
        return out