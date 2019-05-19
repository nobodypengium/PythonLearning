import math
import h5py
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from tensorflow.python.framework import ops
import partIVch1.cnn_utils as cnn_utils
# from partIVch1.network_numpy import *
from partIVch1.network_tf import *
import numpy as np

plt.rcParams['figure.figsize'] = (5.0, 4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(1)

# %%测试两层通道的填充
# x = np.random.randn(4,3,3,2)
# x_paded = zero_pad(x,2)
# print("x.shape = ",x.shape)
# print("x_paded.shape = ",x_paded.shape)
# print("x[1,1] = ",x[1,1])
# print("x_paded[1,1] = ",x_paded[1,1])
#
# fig , axarr = plt.subplots(1,2)
# axarr[0].set_title('x')
# axarr[0].imshow(x[0,:,:,0])
# axarr[1].set_title('x_paded')
# axarr[1].imshow(x_paded[0,:,:,0])

#%% 测试卷积
# a_slice_prev = np.random.randn(4,4,3)
# W = np.random.randn(4,4,3)
# b = np.random.randn(1,1,1)
#
# Z = conv_single_step(a_slice_prev,W,b)
#
# print("Z = "+str(Z))


#%%测试前向传播
# A_prev = np.random.randn(10,4,4,3)
# W = np.random.randn(2,2,3,8)
# b = np.random.randn(1,1,1,8)
##f存在W里
# hparameters={
#     "pad":2,
#     "stride":1
# }
#
# Z,cache_conv = conv_forward(A_prev,W,b,hparameters)
#
# print("np.mean(Z) = " + str(np.mean(Z)))
# print("cache_conv[0][1][2][3] = " + str(cache_conv[0][1][2][3]))

#%% 测试池化

# A_prev = np.random.randn(2,4,4,3)
#
# #p=0
# hparameters = {
#     "f":4,
#     "stride":1
# }
#
# A,cache = pool_forward(A_prev,hparameters,mode="max")
# print("mode max A = " + str(A))
# A,cache = pool_forward(A_prev,hparameters,mode="average")
# print("mode average A = "+str(A))


#%% 测试反向传播
# A_prev = np.random.randn(10,4,4,3)
# W = np.random.randn(2,2,3,8)
# b = np.random.randn(1,1,1,8)
# hparameters = {
#     "pad":2,
#     "stride":1
# }
# #前向传播
# Z,cache_conv = conv_forward(A_prev,W,b,hparameters)
# #反向传播
# dA,dW,db = conv_backward(Z,cache_conv)
# print("dA_mean = " + str(np.mean(dA)))
# print("dW_mean = " + str(np.mean(dW)))
# print("db_mean = " + str(np.mean(db)))

#%% 测试池化的反向传播，不涉及参数的更新但是设计到dA的拓展
# A_prev = np.random.randn(5,5,3,2)
# hparameters = {
#     "stride":1,
#     "f":2
# }
# A,cache = pool_forward(A_prev,hparameters)
# dA = np.random.randn(5,4,2,2)
#
# dA_prev = pool_backward(dA,cache,mode="max")
# print("max:mean"+str(np.mean(dA)))
# print("dA_prev[1,1]"+str(dA_prev[1,1]))
# dA_prev = pool_backward(dA,cache,mode="average")
# print("average:mean"+str(np.mean(dA)))
# print("dA_prev[1,1]"+str(dA_prev[1,1]))




#%%对TF实现的CNN测试

# %%查看数据集
# X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = cnn_utils.load_dataset()
# index = 15
# plt.imshow(X_train_orig[index])
# plt.title("y = " + str(Y_train_orig[:,index]))

# %%测试占位符
# X, Y = create_placehoders(64,64,3,6)

# %%测试初始化参数
# with tf.Session() as sess_test:
#     X,Y = create_placehoders(64,64,3,6)
#     parameters = initialize_parameters()
#     Z3 = forward_propagation(X,parameters)
#
#     init = tf.global_variables_initializer()
#     sess_test.run(init)
#
#     a = sess_test.run(Z3,{X: np.random.randn(2,64,64,3), Y: np.random.randn(2,6)})
#     print("Z3 = " + str(a))
#
#     sess_test.close

# %%测试softmax和计算成本
with tf.Session() as sess_test:
    np.random.seed(1)
    X,Y = create_placehoders(64,64,3,6)
    parameters = initialize_parameters()
    Z3 = forward_propagation(X,parameters)
    cost = compute_cost(Z3,Y)

    init=tf.global_variables_initializer()
    sess_test.run(init)
    a = sess_test.run(cost,{X:np.random.randn(4,64,64,3),Y:np.random.randn(4,6)})
    print("cost = " + str(a))

    sess_test.close()

