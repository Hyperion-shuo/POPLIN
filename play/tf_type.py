import tensorflow as tf

t = tf.constant(2, dtype=tf.int32)
t_float = tf.cast(t, dtype=tf.float32)
gamma = tf.constant(0.9, dtype=tf.float32)
sy_res = tf.pow(gamma, t_float)

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
res = sess.run(sy_res)
print(res, t)