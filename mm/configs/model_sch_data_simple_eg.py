exp_name = 'simple_eg'
work_dir = f'../experiments/{exp_name}'

# model settings
model = dict(
    type='',
    net=dict(type=''),
    loss_cfg=dict(
        loss=dict()
    ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(),
    # init_cfg=dict(type='Pretrained', checkpoint='')
    init_cfg=dict(type='Xavier')
)


# dataset settings
train_pipeline = [
    dict(type=''),
    dict(type='Collect', keys=['', ''], meta_keys=[''])
]

val_pipeline = [
    dict(type=''),
    dict(type='Collect', keys=['', ''], meta_keys=['']),
]

# test_pipline = []

data = dict(
    workers_per_gpu=1,
    train_dataloader=dict(samples_per_gpu=1, persistent_workers=False),
    val_dataloader=dict(samples_per_gpu=1, persistent_workers=False),
    train=dict(
        type='',
        data_root='',
        datalist_file='../datalist/datalist_train.txt',
        pipeline=train_pipeline),
    val=dict(
        type='',
        data_root='',
        datalist_file='../datalist/datalist_val.txt',
        pipeline=val_pipeline),
    test=dict(
        type='',
        data_root='',
        datalist_file='../datalist/datalist_test.txt',
        pipeline=val_pipeline)
)

# eval metrics
evaluation = dict(interval=1)

# optimizer
optimizer_config = dict(grad_clip=None)
optimizers = dict(type='Adam', lr=1e-4, betas=(0.9, 0.99))

# learning policy
lr_config = dict(policy='Step',
                 warmup='linear',
                 warmup_iters=2,
                 warmup_ratio=0.001,
                 step=[5, 8],
                 gamma=0.5)


runner = dict(type='EpochBasedRunner', max_epochs=12)
# runner = dict(type='IterBasedRunner', max_iters=12)

# runtime setting
checkpoint_config = dict(interval=1)

log_config = dict(interval=2,
                  hooks=[
                      dict(type='TextLoggerHook'),
                      dict(type='TensorboardLoggerHook')])

# dictributed trainning
dist_params = dict(backend='nccl')
resume_from = None
log_level = 'INFO'
load_from = None

workflow = [('train', 1)]


