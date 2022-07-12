# dataset settings
train_pipeline = [
    dict(type=''),
    dict(type='Collect', keys=['', ''], meta_keys=['']),
]
val_pipeline = [
    dict(type=''),
    dict(type='Collect', keys=['', ''], meta_keys=['']),
]
test_pipeline = [
    dict(type=''),
    dict(type='Collect', keys=['', ''], meta_keys=['']),
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type='',
        pipeline=train_pipeline),
    val=dict(
        type='',
        pipeline=val_pipeline),
    test=dict(
        type='',
        pipeline=test_pipeline)
)
evaluation = dict(interval=1)
