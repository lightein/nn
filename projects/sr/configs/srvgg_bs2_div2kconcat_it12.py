_base_ = [
    './srvgg_bs2_div2k_it12.py',
]

exp_name = 'srvgg_div2kconcat'
work_dir = f'../../experiments/{exp_name}'


data = dict(
    train=dict(  # DATASETS register
        type='SRGTDataset',
        data_root='D:/xxd/dataset',
        datalist_file=['../../datalists/datalist_div2k_train.txt', '../../datalists/datalist_div2k_val.txt']
    ),
)
