import numpy as np
import h5py
import matplotlib.pyplot as plt
import partIch4.testCases
from partIch4.dnn_utils import *
import partIch4.lr_utils


# %% 初始化两层网络
def initialize_parameters(n_x, n_h, n_y):
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    parameters = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2
    }

    return parameters


# %% 对于深度的神经网络的节点初始化
def initialize_parameters_deep(layers_dims):  # layers_dims秩为1数组，layers_dims[l]包含第l层节点的数量
    np.random.seed(3)
    parameters = {}
    L = len(layers_dims)

    for l in range(1, L):  # 第0层不用初始化，那是输入
        parameters["W" + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) / np.sqrt(
            layers_dims[l - 1])  # 用除以平方根代替防止梯度消失或梯度爆炸
        parameters["b" + str(l)] = np.zeros((layers_dims[l], 1))#zeros写法的诡异错误，要套两层括号

    return parameters
