# Copyright (c) OpenMMLab. All rights reserved.
import math

import numpy as np
import torch
from torchvision.utils import make_grid


def set_requires_grad(nets, requires_grad=False):
    """Set requires_grad for all the networks.

    Args:
        nets (nn.Module | list[nn.Module]): A list of networks or a single
            network.
        requires_grad (bool): Whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def tensor2img(tensor, out_type=np.uint8, min_max=(0, None)):
    """Convert torch Tensors into image numpy arrays.

    After clamping to (min, max), image values will be normalized to [0, None].

    For different tensor shapes, this function will have different behaviors:

        1. 4D mini-batch Tensor of shape (N x 3/1 x H x W):
            Use `make_grid` to stitch images in the batch dimension, and then
            convert it to numpy array.
        2. 3D Tensor of shape (3/1 x H x W) and 2D Tensor of shape (H x W):
            Directly change to numpy array.

    Args:
        tensor (Tensor | list[Tensor]): Input tensors.
        out_type (numpy type): Output types. If ``np.uint8``, transform outputs
            to uint8 type with range [0, 255]; otherwise, float type with
            range [0, None]. Default: ``np.uint8``.
        min_max (tuple): min and max values for clamp.

    Returns:
        (Tensor | list[Tensor]): 3D ndarray of shape (H x W x C) or 2D ndarray
        of shape (H x W).
    """
    if not (torch.is_tensor(tensor) or (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError(f'tensor or list of tensors expected, got {type(tensor)}')

    if torch.is_tensor(tensor):
        tensor = [tensor]
    result = []
    for _tensor in tensor:
        # Squeeze two times so that:
        # 1. (1, 1, h, w) -> (h, w) or
        # 3. (1, 3, h, w) -> (3, h, w) or
        # 2. (n>1, 3/1, h, w) -> (n>1, 3/1, h, w)
        _tensor = _tensor.squeeze(0).squeeze(0)
        _tensor = _tensor.float().detach().cpu().clamp_(*min_max)
        if None not in min_max:
            _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])
        n_dim = _tensor.dim()
        if n_dim == 4:
            img_np = make_grid(_tensor, nrow=int(math.sqrt(_tensor.size(0))), normalize=False).numpy()
            img_np = np.transpose(img_np, (1, 2, 0))
        elif n_dim == 3:
            img_np = _tensor.numpy()
            img_np = np.transpose(img_np, (1, 2, 0))
        elif n_dim == 2:
            img_np = _tensor.numpy()
        else:
            raise ValueError(f'Only support 4D, 3D or 2D tensor. But received with dimension: {n_dim}')
        if out_type == np.uint8:
            # Unlike MATLAB, numpy.unit8() WILL NOT round by default.
            img_np = np.clip((img_np * 255.0).round(), 0, 255)
        img_np = img_np.astype(out_type)
        result.append(img_np)
    result = result[0] if len(result) == 1 else result
    return result


def img2tensor(imgs, float32=True):
    """Numpy array to tensor.
    1. 3D ndarray of shape (H x W x C) change to (C x H x W)
    2. other shape directly change to Tensor

    Args:
        imgs (list[ndarray] | ndarray): Input images.
        float32 (bool): Whether to change to float32.
    Returns:
        list[tensor] | tensor: Tensor images. If returned results only have
            one element, just return tensor.
    """

    def _totensor(img, float32):
        if img.ndim == 3:
            img = torch.from_numpy(img.transpose(2, 0, 1))
        else:
            img = torch.from_numpy(img)
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, float32) for img in imgs]
    else:
        return _totensor(imgs, float32)
