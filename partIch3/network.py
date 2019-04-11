import numpy as np
import matplotlib.pyplot as plt
from partIch3.testCases import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from partIch3.planar_utils import *


# 定义神经网络结构
# 初始化模型参数
# 循环
#   前向传播
#   计算损失
#   后向传播
#   梯度下降

# %%定义神经网络结构
def layer_size(X, Y):
    n_x = X.shape[0]  # 输入层每个样本有n_x个变量
    n_h = 4  # 隐藏层每个样本有4个变量
    n_y = Y.shape[0]  # 输出层每个样本有n_y个变量，实际上1个吧...

    return n_x, n_h, n_y

# %%初始化模型参数，权重矩阵随机初始化，bias初始化为全0
def initialize_parameters(n_x,n_h,n_y):
    np.random.seed(2)
    W1 = np.random.randn(n_h,n_x)*0.01 #第一层用的权重矩阵，初始化为比较小的值
    b1 = np.zeros((n_h,1))
    W2 = np.random.randn(n_y,n_h)*0.01 #第二层的权重矩阵
    b2 = np.zeros((n_y,1))

    parameters = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2
    }

    return parameters

# %%计算前向传播
def forward_propagation(X,parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    Z1 = np.dot(W1,X)+b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2,X)+b2
    A2 = np.tanh(Z2)

    assert(A2.shape == (1,X.shape[1]))


