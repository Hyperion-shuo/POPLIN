from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import argparse
import pprint
import copy

import sys
os.environ["CUDA_VISIBLE_DEVICES"]='0'
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from dotmap import DotMap

from dmbrl.misc.MBExp import MBExperiment
from dmbrl.controllers.MPC import MPC
from dmbrl.config import create_config
from dmbrl.misc import logger


def main(env, ctrl_type, ctrl_args, overrides, logdir, args):
    ctrl_args = DotMap(**{key: val for (key, val) in ctrl_args})
    cfg = create_config(env, ctrl_type, ctrl_args, overrides, logdir)
    logger.info('\n' + pprint.pformat(cfg))

    # add the part of popsize
    if ctrl_type == "MPC":
        cfg.exp_cfg.exp_cfg.policy = MPC(cfg.ctrl_cfg)

    # cfg.exp_cfg.misc = copy.copy(cfg)
    exp = MBExperiment(cfg.exp_cfg)

    if not os.path.exists(exp.logdir):
        os.makedirs(exp.logdir)
    with open(os.path.join(exp.logdir, "config.txt"), "w") as f:
        f.write(pprint.pformat(cfg.toDict()))

    ##################################debug cem cost#############################
    import dmbrl.env
    import gym
    import tensorflow as tf
    import numpy as np

    env = gym.make('MBRLHalfCheetah-v0')
    A = env.action_space.shape
    print(A)

    def obs_cost_fn(obs):
        return -obs[:, 0]

    def ac_cost_fn(acs):
        if isinstance(acs, np.ndarray):
            return 0.1 * np.sum(np.square(acs), axis=1)
        else:
            return 0.1 * tf.reduce_sum(tf.square(acs), axis=1)


    sess = exp.policy.model.sess
    sess.run(tf.global_variables_initializer())

    obs = env.reset()
    exp.policy.sy_cur_obs.load(obs, sess)
    cur_obs = sess.run(exp.policy.sy_cur_obs)
    print(cur_obs)
    ac_seqs = tf.random.uniform((500, 30 * 6), minval=-1, maxval=1, dtype=tf.float32)
    costs, pred_trajs = exp.policy._compile_cost(ac_seqs, get_pred_trajs=True,
                            cem_type=None, tf_data_dict=None)
    costs, pred_trajs, ac_seqs = sess.run([costs, pred_trajs])

    print(costs, pred_trajs)
    #############################################################################

    # exp.run_experiment()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-env', type=str, required=True,
                        help='Environment name: select from [cartpole, reacher, pusher, halfcheetah]')
    parser.add_argument('-ca', '--ctrl_arg', action='append', nargs=2, default=[],
                        help='Controller arguments, see https://github.com/kchua/handful-of-trials#controller-arguments')
    parser.add_argument('-o', '--override', action='append', nargs=2, default=[],
                        help='Override default parameters, see https://github.com/kchua/handful-of-trials#overrides')
    parser.add_argument('-logdir', type=str, default='log',
                        help='Directory to which results will be logged (default: ./log)')
    parser.add_argument('-e_popsize', type=int, default=500,
                        help='different popsize to use')
    # args = parser.parse_args()
    # debug args POPLINP
    # args = parser.parse_args(" -env gym_cartpole  -logdir /data/ShenShuo/workspace/POPLIN/log/debug\
    #         -o exp_cfg.exp_cfg.ntrain_iters 50 \
    #         -o ctrl_cfg.cem_cfg.cem_type POPLINA-REPLAN \
    #         -o ctrl_cfg.cem_cfg.training_scheme BC-AI \
    #         -o ctrl_cfg.cem_cfg.test_policy 0 \
    #         -ca model-type PE -ca prop-type E \
    #         -ca opt-type POPLIN-A".split())

    # debug args PETS 
    args = parser.parse_args(" -env halfcheetah  -logdir /data/ShenShuo/workspace/POPLIN/log/debug\
            -o exp_cfg.exp_cfg.ntrain_iters 50 \
            -ca model-type DE -ca prop-type E \
            -ca opt-type CEM".split())

    main(args.env, "MPC", args.ctrl_arg, args.override, args.logdir, args)
