import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.io as sio
import glob
import os
import pandas as pd

envs = ['CartPole-v0']
algos = ['rtg', 'simple_pg']
data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/*/*/*.mat')
files = glob.glob(data_path)

sns.set(style='darkgrid', font_scale=1.5)

for env in envs:
    for algo in algos:
        data_list = []
        for seed in range(10):
            plot_files = [file for file in files if env in file]
            plot_files = [file for file in plot_files if algo in file]
            plot_files = [file for file in plot_files if str(seed) in file]
            data = sio.loadmat(file_name=plot_files[0])
            data_list.append(np.array(data['return'][0]).reshape(1,-1))
        data_list = np.concatenate(data_list, axis=0)
        # data_list = pd.DataFrame(data_list)
        sns.tsplot(data_list)
    plt.xlabel('epoch')
    plt.ylabel('mean_return')
    plt.title(env)
    plt.savefig(env)
    plt.show()