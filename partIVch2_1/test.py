import tensorflow as tf
import numpy as np
from partIVch2_1.network import *
# import keras.backend as K

# import keras.backend as K #基本的类型转换啊之类的底层（相对于层）的操作
# K.set_image_data_format('channels_last')
# K.set_learning_phase(1) #测试模式

# with tf.Session() as test:
#     np.random.seed(1)
#     A_prev = tf.placeholder('float',[3,4,4,6]) #创建一个tensor变量
#     X = np.random.randn(3,4,4,6)
#     A = identity_block(A_prev,f=2,filters=[2,4,6],stage=1,block="a")
#
#     test.run(tf.global_variables_initializer())
#     out = test.run([A],feed_dict={A_prev:X,K.learning_phase():0})
#     print("out = " + str(out[0][1][1][0]))
#
#     test.close()

# %%测试恒等快
# tf.reset_default_graph()
# with tf.Session() as test:
#     np.random.seed(1)
#     A_prev = tf.placeholder("float",[3,4,4,6])
#     X = np.random.randn(3,4,4,6)
#     A = identity_block(A_prev,f=2,filters=[2,4,6],stage=1,block="a")
#
#     test.run(tf.global_variables_initializer())
#     out = test.run([A], feed_dict={A_prev:X, K.learning_phase():0})
#     print("out = " + str(out[0][1][1][0]))
#
#     test.close()

# %%测试卷积块
tf.reset_default_graph()

with tf.Session() as test:
    np.random.seed(1)
    A_prev = tf.placeholder("float",[3,4,4,6])
    X = np.random.randn(3,4,4,6)

    A = convolutional_block(A_prev, f=2, filters = [2,4,6], stage=1, block="a")

    test.run(tf.global_variables_initializer())
    out = test.run([A],feed_dict={A_prev:X,K.learning_phase():0})
    print("out = " + str(out[0][1][1][0]))

    test.close()

tf.reset_default_graph()