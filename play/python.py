import tensorflow as tf
import numpy as np


gamma = 0.99
plan_hor = 10

discount_seqs = np.ones(plan_hor + 1)
for i in range(1, plan_hor + 1, 1):
    discount_seqs[i] = discount_seqs[i - 1] * gamma

discount_seqs = tf.constant(discount_seqs)

with tf.Session() as sess:
    out_put = sess.run(discount_seqs)
print(out_put)
