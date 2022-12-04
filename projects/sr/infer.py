import os

import numpy as np
import torch
from dnn.datasets.pipelines.compose import Compose
from dnn.models.builder import build_model
from dns.utils.util_raw import rgbg2bayer, simple_aiisp_post


def main():

    exp_name = ''
    model_path = f'../../experiments/dns/{exp_name}/ckpt/{exp_name}/iter_1_st.pth'

    data_info = dict()

    data_info['gt_path'] = r''
    data_info['lq_path'] = r''

    train_pipeline = [
        dict(type='RawIsoGTLQMaskNight1110',
             pipeline_cfg=dict(),
        dict(type='Collect', keys=['lq', 'gt'], meta_keys=['gt_path', 'lq_path'])
    ]

    train_pipeline = [
        dict(type='RawIsoGTLQMaskPrefetch',
             pipeline_cfg=dict(),
        dict(type='Collect', keys=['lq', 'gt'], meta_keys=['gt_path', 'lq_path'])
    ]

    pipline = Compose(train_pipeline)
    data = pipline(data_info)

    data['lq'] = data['lq'].unsqueeze(0)
    data['gt'] = data['gt'].unsqueeze(0)

    model_cfg = dict(
        type='',
        net=dict(type=''),
        loss_cfg=dict(
            loss_l1=dict(type='L1Loss', loss_weight=1.0),
            ),

        # model training and testing settings
        train_cfg=dict(),
        test_cfg=dict(),
        init_cfg=dict(type='Pretrained', checkpoint=model_path)
    )

    model = build_model(model_cfg)

    with torch.no_grad():
        result = model(data, return_loss=False)

    result = result[0]

    save_dir = os.path.join(f'../../experiments/dns/{exp_name}', 'test')
    os.makedirs(save_dir, exist_ok=True)
    save_name = os.path.basename(model_path)[:-4]
    for key, val in result['data_result'].items():

        data_result = val


if __name__ == '__main__':
    main()

