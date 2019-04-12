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
def initialize_parameters(n_x, n_h, n_y):
    np.random.seed(2)
    W1 = np.random.randn(n_h, n_x) * 0.01  # 第一层用的权重矩阵，初始化为比较小的值
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01  # 第二层的权重矩阵
    b2 = np.zeros((n_y, 1))

    parameters = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2
    }

    return parameters


# %%计算前向传播
def forward_propagation(X, parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)  # 注意输出层别用tanh出来个负数怎么log啊

    assert (A2.shape == (1, X.shape[1]))

    cache = {
        "Z1": Z1,
        "A1": A1,
        "Z2": Z2,
        "A2": A2,
    }

    return cache


# %%计算损失
def compute_cost(A2, Y):
    m = Y.shape[1]

    T = np.log(A2)
    cost = -(1 / m) * np.sum(np.multiply(Y, np.log(A2)) + np.multiply((1 - Y), np.log(1 - A2)))
    cost = float(np.squeeze(cost))

    return cost


# %%反向传播
def backward_propagation(parameters, cache, X, Y):
    m = X.shape[1]
    W1, b1, W2, b2 = parameters["W1"], parameters["b1"], parameters["W2"], parameters["b2"]
    Z1, A1, Z2, A2 = cache["Z1"], cache["A1"], cache["Z2"], cache["A2"]

    dZ2 = A2 - Y
    dW2 = (1 / m) * np.dot(dZ2, A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.multiply(np.dot(W2.T, dZ2), 1 - np.power(A1, 2))
    dW1 = (1 / m) * np.dot(dZ1, X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

    grads = {
        "dW1": dW1,
        "db1": db1,
        "dW2": dW2,
        "db2": db2
    }

    return grads

# %%梯度下降
def update_parameters(parameters, grads, learning_rate):
    W1,b1,W2,b2 = parameters["W1"], parameters["b2"], parameters["W2"], parameters["b2"]
    dW1,db1,dW2,db2 = grads["dW1"], grads["db1"], grads["dW2"], grads["db2"]

    W1 = W1 - dW1*learning_rate
    b1 = b1 - db1*learning_rate
    W2 = W2 - dW2*learning_rate
    b2 = b2 - db2*learning_rate

    parameters = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2
    }
    return parameters

# %%整合神经网络
def nn_model(X,Y,n_h,num_iterations,print_cost = False):
    np.random.seed(3)
    n_x,n_y = layer_size(X,Y)[0],layer_size(X,Y)[2]

    parameters = initialize_parameters(n_x,n_h,n_y)

    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    for i in range(num_iterations):
        cache = forward_propagation(X,parameters)
        cost = compute_cost(cache["A2"],Y)
        grads = backward_propagation(parameters,cache,X,Y)
        parameters = update_parameters(parameters,grads,1.2)

        if print_cost:
            if i%100 == 0:
                print("第{0:d}次迭代的成本是：{1:n}".format(i,cost))

    return parameters

# %%预测函数
def predict(parameters, X):
    cache = forward_propagation(X,parameters)
    predictions = np.round(cache["A2"])

    return predictions