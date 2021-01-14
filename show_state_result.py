import glob
import matplotlib as mpl
mpl.use('Agg')
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
# file_list_state = glob.glob(os.path.join(os.path.dirname(__file__),'log/test_env/halfcheetah/*/states.mat'))
file_list_state = glob.glob(os.path.join(os.path.dirname(__file__),'log/test_env/halfcheetah/2021-01-14--14:33:28/iter_state/40states.mat'))
print(file_list_state)

for name in file_list_state:
    true_states = loadmat(name)['t_state']
    pred_states = loadmat(name)['p_state']
    loss = []
    horizon_list = []
    horizon = 20
    while horizon <= true_states.shape[0]:
        true_states_horizon = true_states[ :horizon, :]
        pred_states_horizon = pred_states[ :horizon, :]
        mse = np.mean(np.abs(true_states_horizon - pred_states_horizon))
        loss.append(mse)
        horizon_list.append(horizon)
        horizon += 20
    plt.plot(horizon_list, loss)
    plt.savefig("/data/YanSen/workspace/POPLIN/log/test_env/halfcheetah/2021-01-14--14:33:28/iter_state/iter_40_states.png")
    plt.show()  

