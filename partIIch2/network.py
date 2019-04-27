import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import math
import sklearn
import sklearn.datasets

import partIIch2.opt_utils as opt_utils
import partIIch2.testCase as testCase

plt.rcParams['figure.figsize'] = (7.0, 4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = "gray"


def update_parameters_with_gd(parameters, grads, learning_rate):
    """
    使用普通的梯度下降batch-gradient-decent更新参数
    :param parameters:
    :param grads:
    :param learning_rate:
    :return:
    """
    L = len(parameters) // 2
    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]

    return parameters


def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    """
    乱序、分割
    :param X:
    :param Y:
    :param mini_batch_size64:
    :param seed:
    :return:
    """
    np.random.seed(seed)
    m = X.shape[1]
    mini_batches = []

    # 打乱顺序
    permutation = np.random.permutation(m)  # 0到m-1随机数组，秩为1的矩阵(ndarray类型)，加list命令转换为list类型
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation]

    # 分割
    num_complete_minibatch = math.floor(m / mini_batch_size)  # 返回小于等于x的最大整数
    for k in range(0, num_complete_minibatch):
        mini_batch_X = shuffled_X[:, k * mini_batch_size:(k + 1) * mini_batch_size]  # [横轴全部保留,纵轴保留范围起始:纵轴保留范围结束]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size:(k + 1) * mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, mini_batch_size * num_complete_minibatch:]
        mini_batch_Y = shuffled_Y[:, mini_batch_size * num_complete_minibatch:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


def initialize_velocity(parameters):
    """
    初始化速度
    :param parameters:
    :return:
    """
    L = len(parameters) // 2
    v = {}

    for l in range(L):
        v["dW" + str(l + 1)] = np.zeros_like(parameters["W" + str(l + 1)])  # 生成与Wl同维度全零矩阵
        v["db" + str(l + 1)] = np.zeros_like(parameters["b" + str(l + 1)])

    return v


def update_parameters_with_momentum(parameters, grads, v, beta, learning_rate):
    """
    使用动量梯度下降法更新参数
    :param parameters:之前的参数
    :param grads:梯度
    :param v:平均后的梯度（速度）
    :param beta:控制本次速度和之前速度影响大小的超参数
    :param learning_rage:学习率
    :return:
    """
    L = len(parameters) // 2
    for l in range(L):
        v["dW" + str(l + 1)] = beta * v["dW" + str(l + 1)] + (1 - beta) * grads["dW" + str(l + 1)]
        v["db" + str(l + 1)] = beta * v["db" + str(l + 1)] + (1 - beta) * grads["db" + str(l + 1)]
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * v["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * v["db" + str(l + 1)]

    return parameters, v


def initialize_adam(parameters):
    """
    初始化adam算法涉及到的单输
    :param parameters:
    :return:
    """
    L = len(parameters) // 2
    V = {}
    S = {}

    for l in range(L):
        V["dW" + str(l + 1)] = np.zeros_like(parameters["W" + str(l + 1)])
        V["db" + str(l + 1)] = np.zeros_like(parameters["b" + str(l + 1)])
        S["dW" + str(l + 1)] = np.zeros_like(parameters["W" + str(l + 1)])
        S["db" + str(l + 1)] = np.zeros_like(parameters["b" + str(l + 1)])

    return (V, S)


def update_parameters_with_adam(parameters, grads, V, S, t, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
    """
    使用adam更新参数“自适应矩估计”
    :param parameters: 需要被更新的参数
    :param grads: “步长”
    :param V: 速度
    :param S: 步长的平方（？）避免步长过长
    :param t: 第几次迭代
    :param learning_rate: 学习率
    :param beta1: 控制当次速度和前次速度的影响
    :param beta2: 控制当此速度方和前次速度方的影响
    :param epsilon: 防止除0发生
    :return:
    """
    L = len(parameters) // 2
    V_corrected = {}
    S_corrected = {}

    for l in range(L):
        # 第一矩
        V["dW" + str(l + 1)] = beta1 * V["dW" + str(l + 1)] + (1 - beta1) * grads["dW" + str(l + 1)]
        V["db" + str(l + 1)] = beta1 * V["db" + str(l + 1)] + (1 - beta1) * grads["db" + str(l + 1)]
        V_corrected["dW" + str(l + 1)] = V["dW" + str(l + 1)] / (1 - np.power(beta1, t))
        V_corrected["db" + str(l + 1)] = V["db" + str(l + 1)] / (1 - np.power(beta1, t))
        # 第二矩
        S["dW" + str(l + 1)] = beta2 * S["dW" + str(l + 1)] + (1 - beta2) * np.square(grads["dW" + str(l + 1)])
        S["db" + str(l + 1)] = beta2 * S["db" + str(l + 1)] + (1 - beta2) * np.square(grads["db" + str(l + 1)])
        S_corrected["dW" + str(l + 1)] = S["dW" + str(l + 1)] / (1 - np.power(beta2, t))
        S_corrected["db" + str(l + 1)] = S["db" + str(l + 1)] / (1 - np.power(beta2, t))
        # 更新参数
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * (
                V_corrected["dW" + str(l + 1)] / np.sqrt(S_corrected["dW" + str(l + 1)] + epsilon))
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * (
                V_corrected["db" + str(l + 1)] / np.sqrt(S_corrected["db" + str(l + 1)] + epsilon))

    return (parameters, V, S)


def model(X, Y, layers_dims, optimizer, learning_rate=0.0007, mini_batch_size=64, beta=0.9, beta1=0.9, beta2=0.999,
          epsilon=1e-8, num_epochs=10000, print_cost=True, is_plot=True):
    """
    神经网络模型
    :param X: 输入参数
    :param Y: 输出参数
    :param layers_dims: 神经网络层数、节点数信息
    :param optimizer: 优化类型
    :param learning_rate: 学习率
    :param mini_batch_size: 小批量数据集大小
    :param beta: 动量梯度下降法参数
    :param beta1:
    :param beta2:
    :param epsilon:防止除0
    :param num_epochs:num_iterations 在 mini_batch 中的叫法
    :param print_cost:打印
    :param is_plot:打印
    :return:
    """
    L = len(layers_dims)
    costs = []
    t = 0
    seed = 10

    # 初始化参数W,b，并根据优化方式选择性初始化速度或速度和均方根
    parameters = opt_utils.initialize_parameters(layers_dims)

    # 如果用到了优化那么就有多余测参数需要初始化
    if optimizer == "gd":
        pass
    elif optimizer == "momentum":
        V = initialize_velocity(parameters)
    elif optimizer == "adam":
        V, S = initialize_adam(parameters)
    else:
        print("参数错误")
        exit(1)

        # 前向传播->计算成本->反向传播->更新参数
    for i in range(num_epochs):
        seed = seed + 1  # 实际应用每次应该随机种子，使得数据集排序不同
        minibatches = random_mini_batches(X, Y, mini_batch_size, seed)

        for minibatch in minibatches:
            (minibatch_X, minibatch_Y) = minibatch
            # 前向传播
            AL, cache = opt_utils.forward_propagation(minibatch_X, parameters)
            # 计算成本
            cost = opt_utils.compute_cost(AL, minibatch_Y)
            # 反向传播
            grads = opt_utils.backward_propagation(minibatch_X, minibatch_Y, cache)
            # 更新参数
            if optimizer == "gd":
                parameters = update_parameters_with_gd(parameters, grads, learning_rate)
            elif optimizer == "momentum":
                parameters, V = update_parameters_with_momentum(parameters, grads, V, beta, learning_rate)
            elif optimizer == "adam":
                t = t + 1
                parameters, V, S = update_parameters_with_adam(parameters, grads, V, S, t,learning_rate, beta1, beta2,
                                                               epsilon)
        if i % 100 == 0:
            costs.append(cost)
            if (print_cost and i % 1000 == 0):
                print("第" + str(i) + "次遍历整个数据集误差值：" + str(cost))
        # 画图
        if is_plot:
            plt.plot(costs)
            plt.ylabel("cost")
            plt.xlabel("epochs (per 100)")
            plt.title("learning_rate = " + str(learning_rate)+"\nOptimizer = "+optimizer)

    return parameters,costs

def plot_prediction(title,X,Y,parameters):
    plt.title(title)
    axes = plt.gca()
    axes.set_xlim([-1.5, 2.5])
    axes.set_ylim([-1, 1.5])
    opt_utils.plot_decision_boundary(lambda x: opt_utils.predict_dec(parameters, x.T), X, Y)

