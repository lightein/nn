_base_ = [
    './_base_/datasets/dataset_cfg.py',
    './_base_/models/model_cfg.py',
    './_base_/schedules/schedule_cfg.py',
    './_base_/default_runtime.py'
]
# fp16 settings
# fp16 = dict(loss_scale=512.)

custom_imports = dict(
    imports=[''],
    allow_failed_imports=False)

# runtime setting
exp_name = 'base_test'
# exp_name = '{{fileBasenameNoExtension}}'
work_dir = f'../../experiments/{exp_name}'
