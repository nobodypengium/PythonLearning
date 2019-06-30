from partIVch4_1.network import *

with tf.Session() as test:
    tf.set_random_seed(1)
    y_true = (None, None, None)
    y_pred = (tf.random_normal([3,128],mean=6,stddev=0.1,seed=1),
              tf.random_normal([3,128],mean=1,stddev=1,seed=1),
              tf.random_normal([3,128],mean=3,stddev=4,seed=1))
    loss = triplet_loss(y_true,y_pred)
    print("loss="+str(loss.eval()))
