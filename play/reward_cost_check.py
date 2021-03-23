import numpy as np
import tensorflow as tf
import gym
import time

import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append('/data/ShenShuo/workspace/POPLIN/spinningup')
sys.path.append('/data/ShenShuo/workspace/POPLIN')
import dmbrl.env
import mbbl.env

'''
    mbbl five walker env reward function check
    this is gym_cheetah
'''
from mbbl.env.gym_env import walker
env_name = 'gym_cheetah'
env = walker.env(env_name=env_name, rand_seed=1234,
                            misc_info={'reset_type': 'gym'})


def obs_cost_fn(obs):
    """ @brief:
            see mbbl.env.gym_env.walker.py for reward details
    """
    return -obs[:, 8]  # the qvel for the root-x joint

def ac_cost_fn(acs):
    if isinstance(acs, np.ndarray):
        return 0.1 * np.sum(np.square(acs), axis=1)
    else:
        return 0.1 * tf.reduce_sum(tf.square(acs), axis=1)

reward_diffs = []
cum_rew = 0
cum_cost = 0
obs = env.reset()
print("=" * 20)
print('mbbl gym_cheetah env reward check')
for i in range(30):
    act = env.action_space.sample()
    obs, rew, done, _ = env.step(act)
    cost = (obs_cost_fn(obs.reshape(1, -1)) + ac_cost_fn(act.reshape(1, -1)))
    cum_rew += rew
    cum_cost += cost
    print('\treward:%f, \t-cost: %f, \trel_diff:%f' % (rew, -cost, rew + cost))
print('\tcumulative reward: %f, \tcumulative -cost: %f, \tcumulative diff:%f' 
        % (cum_rew, -cum_cost, cum_rew + cum_cost))
##############################################################################################

from mbbl.env.gym_env import walker
env_name = 'gym_swimmer'
env = walker.env(env_name=env_name, rand_seed=1234,
                            misc_info={'reset_type': 'gym'})


def obs_cost_fn(obs):
    """ @brief:
            see mbbl.env.gym_env.walker.py for reward details
    """
    if isinstance(obs, np.ndarray):
        velocity_cost = -obs[:, 3]  # the qvel for the root-x joint
        return velocity_cost
    else:
        velocity_cost = -obs[:, 3]  # the qvel for the root-x joint
        return velocity_cost

def ac_cost_fn(acs):
    if isinstance(acs, np.ndarray):
        return 0.0001 * np.sum(np.square(acs), axis=1)
    else:
        return 0.0001 * tf.reduce_sum(tf.square(acs), axis=1)

reward_diffs = []
cum_rew = 0
cum_cost = 0
obs = env.reset()
print("=" * 20)
print('mbbl env gym_swimmer reward check')
for i in range(30):
    act = env.action_space.sample()
    cost = (obs_cost_fn(obs.reshape(1, -1)) + ac_cost_fn(act.reshape(1, -1)))
    obs, rew, done, _ = env.step(act)
    cum_rew += rew
    cum_cost += cost
    print('\treward:%f, \t-cost: %f, \trel_diff:%f' % (rew, -cost, rew + cost))
print('\tcumulative reward: %f, \tcumulative -cost: %f, \tcumulative diff:%f' 
        % (cum_rew, -cum_cost, cum_rew + cum_cost))

###############################################################################################

'''
    pets three env check
    poplin use its own cartpole and remove pets's origin cartpole
    this is halfcheetah
'''
env = gym.make('MBRLHalfCheetah-v0')
def obs_cost_fn(obs):
    return -obs[:, 0]

def ac_cost_fn(acs):
    if isinstance(acs, np.ndarray):
        return 0.1 * np.sum(np.square(acs), axis=1)
    else:
        return 0.1 * tf.reduce_sum(tf.square(acs), axis=1)

reward_diffs = []
obs = env.reset()
print("=" * 20)
print('pets env reward check')
for i in range(10):
    act = env.action_space.sample()
    obs, rew, done, _ = env.step(act)
    cost = (obs_cost_fn(obs.reshape(1, -1)) + ac_cost_fn(act.reshape(1, -1)))
    print('\treward:%f, \t-cost: %f, \trel_diff:%f' % (rew, -cost, rew + cost))

'''
###########################################################
                    顺序不一样？？？？？
                    obs act
                    next_obs act ??????
                    关键
###########################################################
'''
    