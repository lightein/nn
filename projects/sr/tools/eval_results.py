import os
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
import cv2

import sys
sys.path.extend(['../..', '../../../mmxx', '../../../mmcv-master'])



def show_eval_result():
    exp_name = ''
    eval_folder = f'../../../experiments/dns/{exp_name}/eval/{exp_name}/iter_1'

    eval_result_paths = glob.glob(os.path.join(eval_folder, "*.pkl"))

    for eval_result in eval_result_paths:
        save_dir = os.path.dirname(eval_result)
        save_name = os.path.basename(eval_result)[:-4]
        with open(eval_result, 'rb') as f:
            result = pickle.load(f)
            for key, val in result['data_result'].items():

                data_result = val




if __name__ == '__main__':
    show_eval_result()
