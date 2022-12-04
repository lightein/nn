import torch
import re
import os
from collections import OrderedDict


def ext_state_dict():
    exp_name = 'srvgggan_bs2_div2k_it12'
    model_path = f'../../../experiments/sr/{exp_name}/ckpt/{exp_name}/iter_1.pth'
    revise_keys = [(r'^generator\.', '')]
    save_keys = 'generator'
    save_dir = os.path.dirname(model_path)
    save_name = os.path.basename(model_path)[:-4]
    checkpoint = torch.load(model_path)
    # get state_dict from checkpoint
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # strip prefix of state_dict
    metadata = getattr(state_dict, '_metadata', OrderedDict())

    for p, r in revise_keys:
        state_dict = OrderedDict(
            {re.sub(p, r, k): v
             for k, v in state_dict.items() if save_keys in k})
    # Keep metadata in state_dict
    # state_dict._metadata = metadata
    torch.save(state_dict, os.path.join(save_dir, f'{save_name}_st.pth'), _use_new_zipfile_serialization=False)


if __name__ == '__main__':
    ext_state_dict()
