import os

import numpy as np
import torch
import cv2

from dnn.datasets.pipelines.compose import Compose
from dnn.models.builder import build_model

import projects.sr


def main():

    exp_name = 'srvgggan_bs2_div2k_it12'
    model_path = f'../../experiments/sr/{exp_name}/ckpt/{exp_name}/iter_1_st.pth'

    data_info = dict()

    data_info['gt_path'] = r'D:\xxd\work\pycharmwork\nn\experiments\sr\srvgggan_bs2_div2k_it12\eval\srvgggan_bs2_div2k_it12\iter_1\000000_gt.png'
    data_info['lq_path'] = r'D:\xxd\work\pycharmwork\nn\experiments\sr\srvgggan_bs2_div2k_it12\eval\srvgggan_bs2_div2k_it12\iter_1\000000_lq.png'

    upscale = 2
    train_pipeline = [
        dict(type='SRPipeline',
             pipeline_cfg=dict(crop_size=(540, 720), scale=upscale)),  # PIPELINES register
        dict(type='ImageToTensor', keys=['gt_img', 'lq_img']),
        dict(type='Collect', keys=['gt_img', 'lq_img'], meta_keys=['gt_path', 'filename'])  # PIPELINES register
    ]

    pipline = Compose(train_pipeline)
    data = pipline(data_info)

    data['lq_img'] = data['lq_img'].unsqueeze(0)
    data['gt_img'] = data['gt_img'].unsqueeze(0)

    model_cfg = dict(
        type='SRGanModel',  # MODELS register
        # MODELS register's Args
        net=dict(
            g=dict(type='SRVGGNetCompact',
                   num_in_ch=3,
                   num_out_ch=3,
                   num_feat=16,
                   num_conv=4,
                   upscale=upscale,
                   act_type='prelu'),  # NETS register
            d=dict(type='UNetDiscriminatorWithSpectralNorm', in_channels=3)
        ),
        loss_cfg=dict(
            loss_l1=dict(type='CharbonnierLoss'),  # LOSS or LOSSES register
            loss_gan=dict(type='GANLoss', gan_type='vanilla', real_label_val=1.0, fake_label_val=0.0, loss_weight=0.01)
        ),
        # model training and testing settings
        train_cfg=dict(disc_steps=1, disc_init_steps=2),
        test_cfg=dict(),
        # init_cfg=dict(type='Pretrained', checkpoint='')
        init_cfg=dict(
            g=dict(type='Pretrained', checkpoint=model_path),
            d=dict(type='Xavier')
        )
    )

    model = build_model(model_cfg)

    with torch.no_grad():
        result = model(data, return_loss=False)

    result = result[0]

    save_dir = os.path.join(f'../../experiments/sr/{exp_name}', 'test')
    os.makedirs(save_dir, exist_ok=True)
    save_name = os.path.basename(model_path)[:-4]
    for key, val in result['data_result'].items():
        data_result = val
        data_result = np.clip(data_result * 255, 0, 255).astype('uint8')
        cv2.imwrite(os.path.join(save_dir, f'{save_name}_{key}.png'), data_result)


if __name__ == '__main__':
    main()

