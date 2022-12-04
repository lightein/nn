import random

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy import special


def filter2D(img, kernel):
    """PyTorch version of cv2.filter2D
    Args:
        img (Tensor): (b, c, h, w)
        kernel (Tensor): (b, k, k)
    """
    k = kernel.size(-1)
    b, c, h, w = img.size()
    if k % 2 == 1:
        img = F.pad(img, (k // 2, k // 2, k // 2, k // 2), mode='reflect')
    else:
        raise ValueError('Wrong kernel size')

    ph, pw = img.size()[-2:]

    if kernel.size(0) == 1:
        # apply the same kernel to all batch images
        img = img.view(b * c, 1, ph, pw)
        kernel = kernel.view(1, 1, k, k)
        return F.conv2d(img, kernel, padding=0).view(b, c, h, w)
    else:
        img = img.view(1, b * c, ph, pw)
        kernel = kernel.view(b, 1, k, k).repeat(1, c, 1, 1).view(b * c, 1, k, k)
        return F.conv2d(img, kernel, groups=b * c).view(b, c, h, w)


def usm_sharp(img, weight=0.5, radius=50, threshold=10):
    """USM sharpening.
    Input image: I; Blurry image: B.
    1. sharp = I + weight * (I - B)
    2. Mask = 1 if abs(I - B) > threshold, else: 0
    3. Blur mask:
    4. Out = Mask * sharp + (1 - Mask) * I
    Args:
        img (Numpy array): Input image, HWC, BGR; float32, [0, 1].
        weight (float): Sharp weight. Default: 1.
        radius (float): Kernel size of Gaussian blur. Default: 50.
        threshold (int):
    """
    if radius % 2 == 0:
        radius += 1
    blur = cv2.GaussianBlur(img, (radius, radius), 0)
    residual = img - blur
    mask = np.abs(residual) * 255 > threshold
    mask = mask.astype('float32')
    soft_mask = cv2.GaussianBlur(mask, (radius, radius), 0)

    sharp = img + weight * residual
    sharp = np.clip(sharp, 0, 1)
    return soft_mask * sharp + (1 - soft_mask) * img


class USMSharp(torch.nn.Module):

    def __init__(self, radius=50, sigma=0):
        super(USMSharp, self).__init__()
        if radius % 2 == 0:
            radius += 1
        self.radius = radius
        kernel = cv2.getGaussianKernel(radius, sigma)
        kernel = torch.FloatTensor(np.dot(kernel, kernel.transpose())).unsqueeze_(0)
        self.register_buffer('kernel', kernel)

    def forward(self, img, weight=0.5, threshold=10):
        blur = filter2D(img, self.kernel)
        residual = img - blur

        mask = torch.abs(residual) * 255 > threshold
        mask = mask.float()
        soft_mask = filter2D(mask, self.kernel)
        sharp = img + weight * residual
        sharp = torch.clip(sharp, 0, 1)
        return soft_mask * sharp + (1 - soft_mask) * img


def circular_lowpass_kernel(cutoff, kernel_size, pad_to=0):
    """2D sinc filter, ref: https://dsp.stackexchange.com/questions/58301/2-d-circularly-symmetric-low-pass-filter
    Args:
        cutoff (float): cutoff frequency in radians (pi is max)
        kernel_size (int): horizontal and vertical size, must be odd.
        pad_to (int): pad kernel size to desired size, must be odd or zero.
    """
    assert kernel_size % 2 == 1, 'Kernel size must be an odd number.'
    kernel = np.fromfunction(
        lambda x, y: cutoff * special.j1(cutoff * np.sqrt(
            (x - (kernel_size - 1) / 2)**2 + (y - (kernel_size - 1) / 2)**2)) / (2 * np.pi * np.sqrt(
                (x - (kernel_size - 1) / 2)**2 + (y - (kernel_size - 1) / 2)**2)), [kernel_size, kernel_size])
    kernel[(kernel_size - 1) // 2, (kernel_size - 1) // 2] = cutoff**2 / (4 * np.pi)
    kernel = kernel / np.sum(kernel)
    if pad_to > kernel_size:
        pad_size = (pad_to - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))
    return kernel


class ShiftStack(nn.Module):
    """
    Shift n-dim tensor in a local window and generate a stacked
    (n+1)-dim tensor with shape (*orig_shapes, wx*wy), where wx
    and wy are width and height of the window
    """
    def __init__(self, window_size):
        # :param window_size: Int or Tuple(Int, Int) in (win_width, win_height) order
        super().__init__()
        wx, wy = window_size if isinstance(window_size, (list, tuple)) else (window_size, window_size)
        assert wx % 2 == 1 and wy % 2 == 1, 'window size must be odd'
        self.rx, self.ry = wx // 2, wy // 2

    def forward(self, tensor):
        # :param tensor: torch.Tensor(N, C, H, W, ...)
        # :return: torch.Tensor(N, C, H, W, ..., wx*wy)
        shifted_tensors = []
        for x_shift in range(-self.rx, self.rx + 1):
            for y_shift in range(-self.ry, self.ry + 1):
                shifted_tensors.append(torch.roll(tensor, shifts=(y_shift, x_shift), dims=(2, 3)))

        return torch.stack(shifted_tensors, dim=-1)


class BoxFilter(nn.Module):
    def __init__(self, window_size, reduction='mean'):
        # :param window_size: Int or Tuple(Int, Int) in (win_width, win_height) order
        # :param reduction: 'mean' | 'sum'
        super().__init__()
        wx, wy = window_size if isinstance(window_size, (list, tuple)) else (window_size, window_size)
        assert wx % 2 == 1 and wy % 2 == 1, 'window size must be odd'
        self.rx, self.ry = wx // 2, wy // 2
        self.area = wx * wy
        self.reduction = reduction

    def forward(self, tensor):
        # :param tensor: torch.Tensor(N, C, H, W, ...)
        # :return: torch.Tensor(N, C, H, W, ...)
        local_sum = torch.zeros_like(tensor)
        for x_shift in range(-self.rx, self.rx + 1):
            for y_shift in range(-self.ry, self.ry + 1):
                local_sum += torch.roll(tensor, shifts=(y_shift, x_shift), dims=(2, 3))

        return local_sum if self.reduction == 'sum' else local_sum / self.area


class NonLocalMeans(nn.Module):
    def __init__(self, h0, search_window_size=11, patch_size=5):
        super().__init__()
        self.h = h0
        self.gen_window_stack = ShiftStack(window_size=search_window_size)
        self.box_sum = BoxFilter(window_size=patch_size, reduction='sum')

    def forward(self, raw_rgb):
        y = torch.mean(raw_rgb, 1, keepdim=True)  # (N, 1, H, W)

        rgb_window_stack = self.gen_window_stack(raw_rgb)  # (N, 4, H, W, wx*wy)
        y_window_stack = self.gen_window_stack(y)  # (N, 1, H, W, wx*wy)

        distances = torch.sqrt(self.box_sum((y.unsqueeze(-1) - y_window_stack) ** 2))  # (N, 1, H, W, wx*wy)
        weights = torch.exp(-distances / (self.h + 1e-12))  # (N, 1, H, W, wx*wy)

        denoised_rgb = (weights * rgb_window_stack).sum(dim=-1) / weights.sum(dim=-1)  # (N, 3, H, W)

        return torch.clamp(denoised_rgb, 0, 1)  # (N, 3, H, W)


if __name__ == '__main__':
    kernel_range = [2 * v + 1 for v in range(3, 11)]  # [7, 21]
    kernel_size = random.choice(kernel_range)
    if kernel_size < 13:
        omega_c = np.random.uniform(np.pi / 3, np.pi)
    else:
        omega_c = np.random.uniform(np.pi / 5, np.pi)
    circular_lowpass_kernel(omega_c, kernel_size)
