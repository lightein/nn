# model settings
model = dict(
    type='',
    net=dict(type=''),
    loss_cfg=dict(
        loss1=dict(type=''),
        # loss2=dict(type='')
    ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(),
    # init_cfg=dict(type='Pretrained', checkpoint='')
    init_cfg=dict(type='Xavier')
)
