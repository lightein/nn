# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.cnn import build_model_from_cfg
from mmcv.utils import Registry

MODELS = Registry('model')
NETS = Registry('net')
LOSSES = Registry('loss')
BACKBONES = Registry('backbones')


def build_backbone(cfg, train_cfg=None, test_cfg=None):
    """Build model.

    Args:
        cfg (dict): Configuration for building model.
        train_cfg (dict): Training configuration. Default: None.
        test_cfg (dict): Testing configuration. Default: None.
    """
    return BACKBONES.build(cfg)


def build_net(cfg, train_cfg=None, test_cfg=None):
    """Build model.

    Args:
        cfg (dict): Configuration for building model.
        train_cfg (dict): Training configuration. Default: None.
        test_cfg (dict): Testing configuration. Default: None.
    """
    return NETS.build(cfg)


def build_loss(cfg):
    """Build loss.

    Args:
        cfg (dict): Configuration for building loss.
    """
    return LOSSES.build(cfg)


def build_losses(cfgs):
    """Build multiple losses from configs.

    If `cfgs` contains several dicts for losses, then a dict for each
    constructed losses will be returned.
    If `cfgs` only contains one loss config, the constructed loss
    itself will be returned.

    For example,

    1) Multiple loss configs:

    .. code-block:: python

        loss_cfg = dict(
            loss_l1=dict(type='L1Loss', loss_weight=0.5, reduction='mean'),
            loss_l2=dict(type='MSELoss', loss_weight=0.5, reduction='mean')
        )

    The return dict is
    ``dict('loss_l1': L1Loss, 'loss_l2': MSELoss)``

    2) Single loss config:

    .. code-block:: python

        loss_cfg = dict(type='L1Loss', loss_weight=0.5, reduction='mean')

    The return is ``L1Loss``.

    Args:
        cfgs (dict): The config dict of the loss.

    Returns:
        dict[:obj:`loss`] | :obj:`loss`:
            The initialized losses.
    """
    losses = {}
    # determine whether 'cfgs' has several dicts for losses
    is_dict_of_dict = True
    for key, cfg in cfgs.items():
        if not isinstance(cfg, dict):
            is_dict_of_dict = False

    if is_dict_of_dict:
        for key, cfg in cfgs.items():
            cfg_ = cfg.copy()
            losses[key] = build_loss(cfg_)
        return losses

    return build_loss(cfgs)


def build_model(cfg, train_cfg=None, test_cfg=None):
    """Build model.

    Args:
        cfg (dict): Configuration for building model.
        train_cfg (dict): Training configuration. Default: None.
        test_cfg (dict): Testing configuration. Default: None.
    """
    return build_model_from_cfg(cfg, MODELS, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg))
