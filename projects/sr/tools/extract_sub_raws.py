import os
import sys
import glob
import argparse
from functools import partial

import numpy as np
import pandas as pd
sys.path.extend(['../..', '../../../mmxx', '../../../mmcv-master'])
from mmcv import track_parallel_progress


def task_fun(kword, raw_shape, raw_dtype, data_root, result_root, crop_sz, step, thres_sz, path):
    path = path.replace('\\', '/')
    img_paths = glob.glob(os.path.join(data_root, path, kword))
    for img_path in img_paths:
        _, ext = os.path.splitext(img_path)
        img_name = os.path.basename(img_path)

        save_folder = os.path.join(result_root, path, img_name[:-4])
        save_folder = f'{save_folder}_sub'

        os.makedirs(save_folder, exist_ok=True)

        img = np.fromfile(img_path, dtype=raw_dtype)
        img = img.reshape(raw_shape)

        n_channels = len(img.shape)
        if n_channels == 2:
            h, w = img.shape
        elif n_channels == 3:
            h, w, c = img.shape
        else:
            raise ValueError('Wrong image shape - {}'.format(n_channels))

        h_space = np.arange(0, h - crop_sz + 1, step)
        if h - (h_space[-1] + crop_sz) > thres_sz:
            h_space = np.append(h_space, h - crop_sz)
        w_space = np.arange(0, w - crop_sz + 1, step)
        if w - (w_space[-1] + crop_sz) > thres_sz:
            w_space = np.append(w_space, w - crop_sz)

        index = 0
        for x in h_space:
            for y in w_space:
                index += 1
                if n_channels == 2:
                    crop_img = img[x:x + crop_sz, y:y + crop_sz]
                else:
                    crop_img = img[x:x + crop_sz, y:y + crop_sz, :]
                crop_img = np.ascontiguousarray(crop_img)
                crop_img.tofile(os.path.join(save_folder, img_name.replace(ext, '_s{:03d}'.format(index) + ext)))
    return 'Processing {:s} ...'.format(path)


def main():
    """A multii-thread tool to crop sub imags."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, help='path to input directory', default=r'')
    parser.add_argument('--datalist_file', type=str, help='list file', default='datalist.txt')
    parser.add_argument('--result_root', type=str, help='path to output directory', default=r'')
    parser.add_argument('--crop_sz', type=int, help='square shape', default=256)
    parser.add_argument('--step', type=int, help='stride', default=256)
    parser.add_argument('--thres_sz', type=int, help='exceed threshold size', default=48)
    parser.add_argument('--raw_shape', nargs='+', type=int)
    parser.add_argument('--raw_dtype', type=str, default='uint32')
    parser.add_argument('--kword', type=str, default='*.raw')

    args = parser.parse_args()

    data_root = args.data_root
    datalist_file = args.datalist_file
    result_root = args.result_root

    data_df = pd.read_csv(datalist_file, header=None, comment='#')
    data_list = data_df[0]

    n_thread = 16
    crop_sz = args.crop_sz   # crop size
    step = args.step  # crop stride
    thres_sz = args.thres_sz

    raw_shape = tuple(args.raw_shape)
    raw_dtype = args.raw_dtype
    kword = args.kword

    task_fun_p = partial(task_fun, kword, raw_shape, raw_dtype, data_root, result_root, crop_sz, step, thres_sz)
    track_parallel_progress(task_fun_p, data_list, n_thread)


if __name__ == '__main__':
    main()
