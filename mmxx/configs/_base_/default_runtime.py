# runtime setting
checkpoint_config = dict(interval=1)

log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

custom_hooks = [dict(type='')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
# auto_resume = True
# cudnn_benchmark = True
workflow = [('train', 1)]
# the only difference between [('train', 1), ('val', 1)] and [('train', 1)]
# is that the runner will calculate losses on validation set after each training epoch.
# workflow = [('train', 1), ('val', 1)]

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'

