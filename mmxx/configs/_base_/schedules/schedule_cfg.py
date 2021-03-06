# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
# optimizers = dict(
#     optim1=dict(type='Adam', lr=1e-4, betas=(0.9, 0.99)),
#     optim2=dict(type='Adam', lr=1e-4, betas=(0.9, 0.99))
# )
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=2,
    warmup_ratio=0.001,
    step=[5, 10],
    gamma=0.5
)

# # Default setting for scaling LR automatically
# #   - `enable` means enable scaling LR automatically
# #       or not by default.
# #   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
# auto_scale_lr = dict(enable=False, base_batch_size=16)

runner = dict(type='EpochBasedRunner', max_epochs=12)
# runner = dict(type='IterBasedRunner', max_iters=12)
