import torch

from dnn.models import build_model
from projects.sr import *


def test_model():
    upscale = 2
    model_cfg = dict(
        type='SRModel',  # MODELS register
        # MODELS register's Args
        net=dict(type='SRVGGNetCompact',
                 num_in_ch=3,
                 num_out_ch=3,
                 num_feat=16,
                 num_conv=4,
                 upscale=upscale,
                 act_type='prelu'),  # NETS register
        loss_cfg=dict(
            loss=dict(type='CharbonnierLoss')  # LOSS or LOSSES register
        ),
        # model training and testing settings
        train_cfg=dict(),
        test_cfg=dict(),
        # init_cfg=dict(type='Pretrained', checkpoint='')
        init_cfg=dict(type='Xavier')
    )

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
            loss=dict(type='CharbonnierLoss')  # LOSS or LOSSES register
        ),
        # model training and testing settings
        train_cfg=dict(),
        test_cfg=dict(),
        # init_cfg=dict(type='Pretrained', checkpoint='')
        init_cfg=dict(
            g=dict(type='Xavier'),
            d=dict(type='Xavier'))
    )

    model = build_model(model_cfg)

    data = dict()
    data['gt_img'] = torch.rand((1, 3, 128, 128))
    data['lq_img'] = torch.rand((1, 3, 64, 64))

    model(data)


def main():
    test_model()


if __name__ == '__main__':
    main()
