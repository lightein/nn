# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import functools

import numpy as np
import torch
from mmcv import Config, DictAction

from dnn.models.builder import build_model

try:
    from mmcv.cnn import get_model_complexity_info
except ImportError:
    raise ImportError('Please upgrade mmcv to >0.6.2')


def parse_args():
    parser = argparse.ArgumentParser(description='cal flops of model')
    parser.add_argument('config', help='model config file path')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--using-gpu',
        action='store_true',
        help='loading model to gpu')
    args = parser.parse_args()
    return args


def input_constructor(input_shape, device=None):
    input_list = []

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if isinstance(input_shape[0], tuple):
        for shape in input_shape:
            input_data = torch.rand(shape, device=device)
            input_list.append(input_data)
    else:
        input_data = torch.rand(input_shape, device=device)
        input_list.append(input_data)

    # data is args of 'forward_dummy'
    input_tensor = dict(data=tuple(input_list))
    return input_tensor


def main():

    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    input_shape = cfg.input_shape

    model = build_model(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    if torch.cuda.is_available() and args.using_gpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model.to(device)
    model.eval()

    if hasattr(model, 'forward_dummy'):
        model.forward = model.forward_dummy
    else:
        raise NotImplementedError(
            'FLOPs counter is currently not currently supported with {}'.
            format(model.__class__.__name__))

    input_constructor_p = functools.partial(input_constructor, device=device)

    flops, params = get_model_complexity_info(model, input_shape, print_per_layer_stat=True, input_constructor=input_constructor_p)

    split_line = '=' * 30

    print(f'{split_line}\nInput shape: {input_shape}\n'
          f'Flops: {flops}\nParams: {params}\n{split_line}')
    print('!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify that the '
          'flops computation is correct.')


if __name__ == '__main__':
    main()
