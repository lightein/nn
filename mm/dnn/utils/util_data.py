import matplotlib.pyplot as plt
import pickle


def plt_show(data, fig_name='fig', vmin=None, vmax=None):
    plt.figure(fig_name)
    if vmin is None or vmax is None:
        plt.imshow(data, cmap='jet')
    else:
        plt.imshow(data, cmap='jet', vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.savefig(f'{fig_name}.png')


def pkl_dump(data, pkl_file='data.pkl'):
    with open(pkl_file, 'wb') as f:
        pickle.dump(data, f)


def pkl_load(pkl_file):
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    return data
