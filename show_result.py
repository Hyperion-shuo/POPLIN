import glob
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import os

# file_list = glob.glob('./log/*/*/logs.mat')
# file_list = [name for name in file_list if 'old' not in name]
# file_list = [name for name in file_list if '2500' in name]
# legend_lable = []

# file_list = glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)),'log/*/*/logs.mat'))
file_list = glob.glob(os.path.join(os.path.dirname(__file__),'log/*/*/logs.mat'))
print(file_list)
legend_lable = []

colormap = plt.cm.gist_ncar
# plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, len(file_list))])

for name in file_list:
#     returns = loadmat(name)['returns']
    returns = loadmat(name)['returns']
    print(name + '\n')
    print(returns)
    print('\n\n')
    # import pdb; pdb.set_trace()
    plt.plot(returns.reshape([-1]))
    legend_lable.append(name.split('/')[2])

plt.legend(legend_lable)
plt.savefig("POPLIN_test.jpg")
plt.show()
