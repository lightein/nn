import pickle
import math
import os
import glob

import torch
import matplotlib.pyplot as plt
import numpy as np


def get_data_list(indir, keyword):
    data_list = []
    for root, dirs, files in os.walk(indir):
        dirs.sort()
        path_list = glob.glob(os.path.join(root, keyword))
        if path_list:
            data_list.append(root)
    return data_list


def plt_show(data, fig_name='fig', vmin=None, vmax=None):
    fig = plt.figure()
    fig.suptitle(fig_name)
    plt.imshow(data, cmap='jet', vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.savefig(f'{fig_name}.png')


def plt_plot(data, fig_name='fig'):
    fig = plt.figure()
    fig.suptitle(fig_name)
    plt.plot(data)
    plt.savefig(f'{fig_name}.png')


def pkl_dump(data, pkl_file='data.pkl'):
    with open(pkl_file, 'wb') as f:
        pickle.dump(data, f)


def pkl_load(pkl_file):
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    return data


def cvt_nchw(data, shape_fmt):
    idx_n = shape_fmt.find('N')
    if idx_n == -1:
        idx_n = 0
        shape_fmt = shape_fmt[:idx_n] + 'N' + shape_fmt[idx_n:]
        data = np.expand_dims(data, 0)
    idx_c = shape_fmt.find('C')
    if idx_c == -1:
        idx_c = 1
        shape_fmt = shape_fmt[:idx_c] + 'C' + shape_fmt[idx_c:]
        data = np.expand_dims(data, 1)
    idx_h = shape_fmt.find('H')
    if idx_h == -1:
        idx_h = 2
        shape_fmt = shape_fmt[:idx_h] + 'H' + shape_fmt[idx_h:]
        data = np.expand_dims(data, 2)
    idx_w = shape_fmt.find('W')
    if idx_w == -1:
        idx_w = 3
        shape_fmt = shape_fmt[:idx_w] + 'W' + shape_fmt[idx_w:]
        data = np.expand_dims(data, 3)
    data = np.transpose(data, (idx_n, idx_c, idx_h, idx_w))
    return data


def cal_layout_square(n, c):
    layout_h, layout_w = min(n, c), max(n, c)
    x = n * c
    for i in range(int(math.sqrt(x)), 0, -1):
        if x % i == 0:
            p = i
            q = x // i
            layout_h, layout_w = min(p, q), max(p, q)
            break

    return layout_h, layout_w


def plt_show_tensor(data, fig_name='fig', vmin=None, vmax=None, shape_fmt='NCHW', layout=None, disp=None):
    px = 1 / plt.rcParams["figure.dpi"]  # pixel in inches
    fig = plt.figure(figsize=(1920 * px, 1080 * px))
    if torch.is_tensor(data):
        data = data.detach().cpu().numpy()

    shape_fmt = shape_fmt.upper()
    if shape_fmt != 'NCHW':
        data = cvt_nchw(data, shape_fmt)

    n, c, h, w = data.shape
    data_ = np.split(data.reshape((n * c, h, w)), n * c)

    if layout == "square":
        layout_h, layout_w = cal_layout_square(n, c)
    else:
        layout_h, layout_w = n, c  # original layout

    fig.suptitle(f'{fig_name},{n}x{c}x{h}x{w},{layout_h}x{layout_w}')

    if disp == "split":
        for i in range(1, layout_h * layout_w + 1):
            plt.subplot(layout_h, layout_w, i)
            t1_layout = np.squeeze(data_[i - 1], axis=0)
            plt.imshow(t1_layout, cmap="jet", vmin=vmin, vmax=vmax)
            plt.colorbar()
            plt.title(f'{i}')
    else:
        data_hstack = []
        for i in range(0, n * c, layout_w):
            data_hstack.append(np.concatenate(data_[i: i + layout_w], axis=2))
        data_layout = np.squeeze(np.concatenate(data_hstack, axis=1), axis=0)
        plt.imshow(data_layout, cmap='jet', vmin=vmin, vmax=vmax)
        plt.colorbar()

    plt.show()
    plt.savefig(f'{fig_name}.png')
    # plt.close(fig)
