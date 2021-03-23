import numpy as np
import gym
import time
from gym import utils
from gym.envs.mujoco import mujoco_env


env = gym.make('Hopper-v1')
start_obs = env.reset()

start_state = env.get_state()
sim_state = None

for i in range(500):
    action = env.action_space.sample()
    obs, r, done, _ = env.step(action)
    # sim_state = env.sim.get_state()
    # env.render()
    time.sleep(0.01)

# print('sim_state: \t', sim_state)
# print('start_state: \t', start_state)
#
# env.sim.set_state(start_state)
# env.render()
# set_state = env.sim.get_state()
# print('set_state: \t', set_state)
#
# for i in range(100):
#     action = np.zeros_like(env.action_space.sample())# env.action_space.sample()
#     obs, r, done, _ = env.step(action)
#     env.render()
#     time.sleep(0.01)
#
# print('end simulation')