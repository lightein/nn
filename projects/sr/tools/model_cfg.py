# model settings
model = dict(
    type='SRVGGModel',  # MODELS register
    # MODELS register's Args
    net=dict(type='SRVGGNetCompact',
             num_in_ch=3,
             num_out_ch=3,
             num_feat=16,
             num_conv=4,
             upscale=2,
             act_type='prelu'),  # NETS register
    loss_cfg=dict(
        loss=dict(type='CharbonnierLoss')  # LOSS or LOSSES register
    ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(),
    init_cfg=dict(type='Xavier')
)
input_shape = (1, 3, 1080, 1920)
custom_imports = dict(
    imports=['projects.sr'],
    allow_failed_imports=False)
