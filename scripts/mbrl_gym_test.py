from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import gym
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append('/data/ShenShuo/workspace/POPLIN/spinningup')
sys.path.append('/data/ShenShuo/workspace/POPLIN')

import dmbrl.env
import mbbl.env


from mbbl.env.gym_env import walker
env_name = 'gym_cheetah'
env = walker.env(env_name=env_name, rand_seed=1234,
                            misc_info={'reset_type': 'gym'})
print(env.action_space.shape, env.observation_space.shape)
 
# env = gym.make('MBRLHalfCheetah-v0')
# obs = env.reset()
