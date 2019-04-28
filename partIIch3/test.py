import numpy as np
import h5py
import tensorflow as tf
from tensorflow.python.framework import ops
import partIIch3.tf_utils as tf_utils
import time
from partIIch3.network import *
import matplotlib.pyplot as plt


X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = tf_utils.load_dataset()

# %%测试导入数据
# index = 11
# plt.imshow(X_train_orig[index])
# plt.title("Y = " + str(Y_train_orig[:,index]))

X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0],-1).T
X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0],-1).T

#归一化
X_train = X_train_flatten/255
X_test = X_test_flatten/255

#转化为独热矩阵
Y_train = tf_utils.convert_to_one_hot(Y_train_orig,6)
Y_test = tf_utils.convert_to_one_hot(Y_test_orig,6)

# %% 创建占位符
X, Y = create_placeholder(X_train.shape[0],Y_train.shape[0])
# print("X = " + str(X))
# print("Y = " + str(Y))

# %%测试参数初始化
parameters = initialize_parameters()

# %%测试前向传播
Z3 = forward_propagation(X,parameters)

# %%测试计算成本
cost = compute_cost(Z3,Y)
