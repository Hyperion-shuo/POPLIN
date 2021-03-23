from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import gym

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
import dmbrl.env
env = gym.make('MBRLHalfCheetah-v0')
obs = env.reset()
for i in range(10):
    act = env.action_space.sample()
    obs, reward, done, _ = env.step(act)
    print(obs)