from partIVch4.network import *

#%% 测试内容代价函数
# with tf.Session() as test:
#     tf.set_random_seed(1)
#     a_C=tf.random_normal([1,4,4,3],mean=1,stddev=4)
#     a_G=tf.random_normal([1,4,4,3],mean=1,stddev=4)
#     J_content = compute_content_cost(a_C,a_G)
#     print("J_content = "+str(J_content.eval()))
#     test.close()

#%% 测试计算风格矩阵 自己乘自己的转置
# with tf.Session() as test:
#     tf.set_random_seed(1)
#     A = tf.random_normal([3,2*1],mean=1,stddev=4)
#     GA = gram_matrix(A)
#     print("GA = " + str(GA.eval()))
#     test.close()

#%% 测试计算风格损失

with tf.Session() as test:
    tf.set_random_seed(1)
    a_S = tf.random_normal([1,4,4,3],mean=1,stddev=4)
    a_G = tf.random_normal([1,4,4,3],mean=1,stddev=4)
    J_style_layer = compute_layer_style_cost(a_S,a_G)
    print("J_style_layer = " + str(J_style_layer.eval()))

    test.close()
    