_base_ = [
    './srvgg_bs2_div2k_it12.py',
]

data = dict(
    train=dict(  # DATASETS register
        type='SRGTDataset',
        data_root='D:/xxd/dataset',
        datalist_file=['../../datalists/datalist_div2k_train.txt', '../../datalists/datalist_div2k_val.txt']
    ),
)

exp_name = '{{fileBasenameNoExtension}}'
work_dir = f'../../experiments/{exp_name}'
checkpoint_config = dict(out_dir=f'{work_dir}/ckpt')

evaluation = dict(out_dir=f'{work_dir}/eval')  # eval hook