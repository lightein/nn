import os
import glob
import pickle
import sys

import numpy as np
import cv2

sys.path.extend(['../..', '../../../mmxx', '../../../mmcv-master'])


def show_eval_result():
    exp_name = 'srvgggan_bs2_div2k_it12'
    eval_folder = f'../../../experiments/sr/{exp_name}/eval/{exp_name}/iter_1'

    eval_result_paths = glob.glob(os.path.join(eval_folder, "*.pkl"))

    for eval_result in eval_result_paths:
        save_dir = os.path.dirname(eval_result)
        save_name = os.path.basename(eval_result)[:-4]
        with open(eval_result, 'rb') as f:
            result = pickle.load(f)
            for key, val in result['data_result'].items():
                data_result = val
                data_result = np.clip(data_result*255, 0, 255).astype('uint8')
                cv2.imwrite(os.path.join(save_dir, f'{save_name}_{key}.png'), data_result)


if __name__ == '__main__':
    show_eval_result()
