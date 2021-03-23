import tensorflow as tf
import numpy as np
import gym
from gym.spaces import Box ,Discrete
import argparse
from scipy.io import savemat
import os

def mlp(x, sizes, activation=tf.tanh, output_activation=None):
    for size in sizes[:-1]:
        x = tf.layers.dense(inputs=x, units=size, activation=activation)
    return tf.layers.dense(inputs=x, units=sizes[-1], activation=output_activation)

def reward_to_go(rews):
    n = len(rews)
    rtgs = np.zeros_like(rews)

    # rews_sum = 0
    # for i in reversed(range(n)):
    #     rews_sum += rews[i]
    #     rtgs[i] = rews_sum

    for i in reversed(range(n)):
        rtgs[i] = rews[i] + (rtgs[i+1] if i+1 < n else 0)
    return rtgs



def train(env_name="CartPole-v0", lr=1e-2, hidden_sizes=[32,],
          epochs=50, batch_size=5000, render=False, seed=0):

    np.random.seed(seed)
    tf.set_random_seed(seed)
    env=gym.make(env_name)
    assert isinstance(env.observation_space, Box), \
        "only works for env with continuous observation space"
    assert isinstance(env.action_space, Discrete), \
        'only works for env with discrete action space'

    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n

    print('obs_dim:', obs_dim, ' n_acts:', n_acts)

    ob_ph = tf.placeholder(dtype=tf.float32, shape=(None, obs_dim), name='ob_ph')
    logits = mlp(ob_ph, hidden_sizes+[n_acts])

    # maybe need to squeeze
    # (batch_size, 1) to (batch_size)
    '''
    logits: 2-D Tensor with shape `[batch_size, num_classes]`.  Each slice
      `[i, :]` represents the unnormalized log-probabilities for all classes.
    num_samples: 0-D.  Number of independent samples to draw for each row slice.
    '''
    actions = tf.squeeze(tf.random.categorical(logits, num_samples=1), axis=1)

    weights_ph = tf.placeholder(dtype=tf.float32, shape=(None,), name="weights_ph")
    act_ph = tf.placeholder(dtype=tf.int32, shape=(None,), name="act_ph")
    # tf.one_hot inputs (indices, deepth)
    action_mask = tf.one_hot(act_ph, n_acts)
    log_prob = tf.reduce_sum(action_mask * tf.nn.log_softmax(logits), axis=1)
    # minmize -loss means maximize return
    loss = -tf.reduce_mean(log_prob * weights_ph)

    train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())


    def train_one_epoch():
        batch_obs = []
        batch_acts = []
        batch_weights = []
        batch_rets = []
        batch_lens = []

        obs = env.reset()
        done = False
        ep_rews = []

        finished_rendering_this_epoch = False

        while True:

            if (not finished_rendering_this_epoch) and render:
                env.render()

            batch_obs.append(obs.copy())
            act = sess.run(actions, feed_dict={ob_ph: obs.reshape(1, -1)})[0]
            obs, r, done, _ = env.step(act)

            ep_rews.append(r)
            batch_acts.append(act)

            if done:
                ep_rets, ep_lens = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_rets)
                batch_lens.append(ep_lens)
                batch_weights += list(reward_to_go(ep_rews))

                obs, done, ep_rews = env.reset(), False, []
                finished_rendering_this_epoch = True

                if len(batch_obs) > batch_size: break

        batch_loss, _ = sess.run([loss, train_op],
                                 feed_dict={
                                     ob_ph: batch_obs,
                                     weights_ph: batch_weights,
                                     act_ph: batch_acts
                                 })

        return batch_loss, batch_rets, batch_lens


    dir_name = os.path.join('data', os.path.join(env_name, 'rtg'))
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    mean_rets, mean_len , save_dict = [], [], {}
    for i in range(epochs):
        batch_loss, batch_rets, batch_lens = train_one_epoch()
        mean_rets.append(np.mean(batch_rets))
        mean_len.append(np.mean(batch_lens))
        print("epochs: %3d \t loss %.3f \t batch_rets: %.3f \t batch_lens: %.3f" %
              (i, batch_loss, np.mean(batch_rets), np.mean(batch_lens)))
    save_dict = {'return': mean_rets, 'len': mean_len}
    file_name = os.path.join(dir_name, 's' + str(seed) + '.mat')
    savemat(file_name=file_name, mdict=save_dict)




if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument('--env_name', '--env', type=str, default='CartPole-v0')
    parse.add_argument('--render', action='store_true')
    parse.add_argument('--lr', type=float, default=1e-2)
    parse.add_argument('--num_seeds', '-n', type=int, default=10)
    args = parse.parse_args()
    for seed in range(args.num_seeds):
        print('trainning with seed %d' % seed)
        train(env_name=args.env_name, render=args.render, lr=args.lr, seed=seed)