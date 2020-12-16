# -----------------------------------------------------------------------------
#   @author:
#       Tingwu Wang
# -----------------------------------------------------------------------------

import numpy as np
import tensorflow as tf

# only this three keys
# store their statistics
_ALLOW_KEY = ['state', 'diff_state', 'action']


def init_whitening_stats(key_list):
    whitening_stats = {}
    for key in key_list:
        # step means time steps in MDP
        # initialize as 0.01 to avoid calculate mean = sum / step 's zero step
        whitening_stats[key] = {'mean': 0.0, 'variance': 1, 'step': 0.01,
                                'square_sum': 0.01, 'sum': 0.0, 'std': np.nan}
    return whitening_stats


def update_whitening_stats(whitening_stats, rollout_data, key):
    # collect the info
    new_sum, new_step_sum, new_sq_sum = 0.0, 0.0, 0.0

    if type(rollout_data) is dict:
        # dict conccat all rollout
        # notice dim, sum along axis0
        new_sum += rollout_data[key].sum(axis=0)
        new_sq_sum += (np.square(rollout_data[key])).sum(axis=0)
        new_step_sum += rollout_data[key].shape[0]
    else:
        assert type(rollout_data) is list
        # each rollout is a list
        # each list has a dict inside
        for i_episode in rollout_data:
            if key == 'state':
                i_data = i_episode['obs']
            elif key == 'action':
                i_data = i_episode['actions']
            else:
                assert key == 'diff_state'
                i_data = i_episode['obs'][1:] - i_episode['obs'][:-1]

            new_sum += i_data.sum(axis=0)
            new_sq_sum += (np.square(i_data)).sum(axis=0)
            new_step_sum += i_data.shape[0]

    # update the whitening info
    whitening_stats[key]['step'] += new_step_sum
    whitening_stats[key]['sum'] += new_sum
    whitening_stats[key]['square_sum'] += new_sq_sum
    whitening_stats[key]['mean'] = \
        whitening_stats[key]['sum'] / whitening_stats[key]['step']
    whitening_stats[key]['variance'] = np.maximum(
        whitening_stats[key]['square_sum'] / whitening_stats[key]['step'] -
        np.square(whitening_stats[key]['mean']), 1e-2
    )
    whitening_stats[key]['std'] = \
        (whitening_stats[key]['variance'] + 1e-6) ** .5


def add_whitening_operator(whitening_operator, whitening_variable, name, size):
    '''
        whitening_operator is a dict contain all named operator
        whitening_variable is a list using for counting parameter num and intialize
        see file base policy line 50 _build_ph function for using
    '''
    with tf.variable_scope('whitening_' + name):
        whitening_operator[name + '_mean'] = tf.Variable(
            np.zeros([1, size], np.float32),
            name=name + "_mean", trainable=False
        )
        whitening_operator[name + '_std'] = tf.Variable(
            np.ones([1, size], np.float32),
            name=name + "_std", trainable=False
        )
        whitening_variable.append(whitening_operator[name + '_mean'])
        whitening_variable.append(whitening_operator[name + '_std'])

        # the reset placeholders
        whitening_operator[name + '_mean_ph'] = tf.placeholder(
            tf.float32, shape=(1, size), name=name + '_reset_mean_ph'
        )
        whitening_operator[name + '_std_ph'] = tf.placeholder(
            tf.float32, shape=(1, size), name=name + '_reset_std_ph'
        )

        # the tensorflow operators
        whitening_operator[name + '_mean_op'] = \
            whitening_operator[name + '_mean'].assign(
                whitening_operator[name + '_mean_ph']
        )

        whitening_operator[name + '_std_op'] = \
            whitening_operator[name + '_std'].assign(
                whitening_operator[name + '_std_ph']
        )


def copy_whitening_var(whitening_stats, input_name, output_name):
    '''
        copy input_name in whitening_stats to output_name in whitening_stats
        only mean and std is copyed
    '''
    whitening_stats[output_name] = {}
    whitening_stats[output_name]['mean'] = whitening_stats[input_name]['mean']
    whitening_stats[output_name]['std'] = whitening_stats[input_name]['std']

# ust whitening_operator to assign whitening_stats[key][item] to tf.Variable(name=key+item)
# usually state mean and state std
def set_whitening_var(session, whitening_operator, whitening_stats, key_list):

    for i_key in key_list:
        for i_item in ['mean', 'std']:
            session.run(
                whitening_operator[i_key + '_' + i_item + '_op'],
                feed_dict={whitening_operator[i_key + '_' + i_item + '_ph']:
                            # notice two dim here, first dim 1
                           np.reshape(whitening_stats[i_key][i_item], [1, -1])}
            )


def append_normalized_data_dict(data_dict, whitening_stats,
                                target=['start_state', 'diff_state',
                                        'end_state']):
    '''
        use whitening_stat to normalize data_dict
        target in start_state, end_state, diff_state
        the normalized variable stored in data_dict
    '''
    data_dict['n_start_state'] = \
        (data_dict['start_state'] - whitening_stats['state']['mean']) / \
        whitening_stats['state']['std']
    data_dict['n_end_state'] = \
        (data_dict['end_state'] - whitening_stats['state']['mean']) / \
        whitening_stats['state']['std']
    data_dict['n_diff_state'] = \
        (data_dict['end_state'] - data_dict['start_state'] -
         whitening_stats['diff_state']['mean']) / \
        whitening_stats['diff_state']['std']
    data_dict['diff_state'] = \
        data_dict['end_state'] - data_dict['start_state']

# test case
# import numpy as np

# whitening_stats = init_whitening_stats(['state'])
# print(whitening_stats['state'])
# update_whitening_stats(whitening_stats, {'state': np.arange(10)}, 'state')
# print(whitening_stats['state'])
# update_whitening_stats(whitening_stats, {'state': np.arange(15)}, 'state')
# print(whitening_stats['state'])

# whitening_operator, whitening_variable = dict(), list()
# add_whitening_operator(whitening_operator, whitening_variable, 'state', 8)
# print("whitening_operator: \n\t", whitening_operator, '\n\n')
# print("whitening_variable: \n\t",whitening_variable)