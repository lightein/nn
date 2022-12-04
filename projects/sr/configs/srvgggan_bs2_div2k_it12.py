# model settings
upscale = 2
model = dict(
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
        g=dict(type='Xavier'),
        d=dict(type='Xavier')
    )
)

# dataset settings
train_pipeline = [
    dict(type='SRPipeline',
         pipeline_cfg=dict(crop_size=(240, 240), scale=upscale)),  # PIPELINES register
    dict(type='ImageToTensor', keys=['gt_img', 'lq_img']),
    dict(type='Collect', keys=['gt_img', 'lq_img'], meta_keys=['gt_path', 'filename'])  # PIPELINES register
]
val_pipeline = [
    dict(type='SRPipeline',
         pipeline_cfg=dict(crop_size=(540, 720), scale=upscale)),  # PIPELINES register
    dict(type='ImageToTensor', keys=['gt_img', 'lq_img']),
    dict(type='Collect', keys=['gt_img', 'lq_img'], meta_keys=['gt_path', 'filename']),  # PIPELINES register
]
data = dict(
    train_dataloader=dict(samples_per_gpu=2, workers_per_gpu=1, persistent_workers=False),
    val_dataloader=dict(samples_per_gpu=2, workers_per_gpu=1, persistent_workers=False),
    test_dataloader=dict(),
    train=dict(  # DATASETS register
        # type='RepeatDataset',
        # dataset=dict(
        type='SRGTDataset',
        data_root='D:/xxd/dataset',
        datalist_file='../../datalists/datalist_div2k_train.txt',
        # datalist_file=['../../datalists/datalist_div2k_train.txt', '../../datalists/datalist_div2k_val.txt']
        pipeline=train_pipeline
        # ),
        # times=2,
    ),
    val=dict(  # DATASETS register
        type='SRGTDataset',
        data_root='D:/xxd/dataset',
        datalist_file='../../datalists/datalist_div2k_val.txt',
        pipeline=val_pipeline),
    test=dict()
)

# training schedule
# optimizer
optimizer_config = dict(grad_clip=None)  # optimizer hook

# dict's keys are same as model name
optimizers = dict(
    generator=dict(type='Adam', lr=1e-4, betas=(0.9, 0.99)),
    discriminator=dict(type='Adam', lr=1e-4, betas=(0.9, 0.99)),
)  # optimizers register

# learning policy
lr_config = dict(policy='Step',
                 warmup='linear',
                 warmup_iters=2,
                 warmup_ratio=0.001,
                 step=[5, 8],
                 gamma=0.5)
# runner = dict(type='EpochBasedRunner', max_epochs=12)
runner = dict(type='IterBasedRunner', max_iters=12)

# runtime setting
exp_name = '{{fileBasenameNoExtension}}'
work_dir = f'../../experiments/sr/{exp_name}'
checkpoint_config = dict(interval=2, out_dir=f'{work_dir}/ckpt')
log_config = dict(interval=1, hooks=[dict(type='TextLoggerHook'), dict(type='TensorboardLoggerHook')])
# dictributed trainning
dist_params = dict(backend='nccl')
resume_from = None
log_level = 'INFO'
load_from = None
workflow = [('train', 1)]
# val log will be cleaned by eval hook
# workflow = [('train', 3), ('val', 1)]

# eval metrics
evaluation = dict(interval=1, save_results=True, out_dir=f'{work_dir}/eval')  # eval hook

custom_imports = dict(
    imports=['projects.sr'],
    allow_failed_imports=False)
