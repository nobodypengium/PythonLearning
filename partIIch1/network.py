import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import partIIch1.init_utils as init_utils
import partIIch1.reg_utils as reg_utils
import partIIch1.gc_utils as gc_utils


def initialize_parameters_zeros(layers_dims):
    """
    用全零初始化所有参数，这样会导致结果是输入的线性累加，起不到什么学习效果
    :param layers_dims:
    :return:
    """
    parameters = {}

    L = len(layers_dims)
    for l in range(1, L):
        parameters["W" + str(l)] = np.zeros((layers_dims[l], layers_dims[l - 1]))
        parameters["b" + str(l)] = np.zeros((layers_dims[l], 1))

    return parameters


def initialize_parameters_random(layers_dims):
    """
    随机初始化变量W，但是值比较大，这样会出现梯度爆炸
    :param layers_dims:
    :return:
    """

    np.random.seed(3)
    parameters = {}

    L = len(layers_dims)
    for l in range(1, L):
        parameters["W" + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * 10  # 放大参数，使梯度爆炸更明显
        parameters["b" + str(l)] = np.zeros((layers_dims[l], 1))

    return parameters


def initialize_parameters_he(layers_dims):
    """
    随机初始化变量，并抑制梯度异常，使用方差2/layers_dims[l-1]，这适合ReLU激活函数
    :param layers_dims:
    :return:
    """

    np.random.seed(3)
    parameters = {}

    L = len(layers_dims)
    for l in range(1, L):
        parameters["W" + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * np.sqrt(
            2 / layers_dims[l - 1])  # 适应ReLU的初始化权值的方差
        parameters["b" + str(l)] = np.zeros((layers_dims[l], 1))

    return parameters


def model(X, Y, learning_rate=0.01, num_iterations=15000, print_cost=True, initialization="he"):
    """
    先用源码提供的通用工具包试着写一下，没问题再换自己的
    :param X:
    :param Y:
    :param learning_rate:
    :param num_iterations:
    :param print_cost:
    :param initialization:
    :param isPlot:
    :return:
    """
    grads = {}  # 最后一次的梯度，只是记录，对于实际模型没有作用
    costs = []  # 成本函数，只是记录用于输出，对实际模型没有作用

    m = X.shape[1]
    layer_dims = [X.shape[0], 10, 5, 1]

    # 选择初始化参数的类型
    if initialization == "zeros":
        parameters = initialize_parameters_zeros(layer_dims)
    elif initialization == "random":
        parameters = initialize_parameters_random(layer_dims)
    elif initialization == "he":
        parameters = initialize_parameters_he(layer_dims)
    else:
        print("错误的初始化参数，退出！")
        exit

    # 迭代
    for i in range(0, num_iterations):
        # 前向传播
        AL, caches = init_utils.forward_propagation(X, parameters)
        # 计算成本
        cost = init_utils.compute_loss(AL, Y)
        # 反向传播
        grads = init_utils.backward_propagation(X, Y, caches)
        # 更新参数
        parameters = init_utils.update_parameters(parameters, grads, learning_rate)

        # 记录成本
        if i % 1000 == 0:
            costs.append(cost)
            if print_cost:
                print("第" + str(i) + "次迭代的成本是" + str(cost))

    return parameters, costs

def model_reg(X,Y,learning_rate=0.3,num_iteration=30000,print_cost=True,lambd=0,keep_prob=1):
    """
    用于测试是否正则化对方差（过拟合与否）的影响
    :param X:
    :param Y:
    :param learning_rate:
    :param num_iteration:
    :param print_cost:
    :param lambd:
    :param keep_prob:
    :return:
    """
    # 缓存变量
    grads={}
    costs=[]
    m = X.shape[1]
    layers_dims = [X.shape[0],20,3,1]

    # 初始化参数
    parameters = reg_utils.initialize_parameters(layers_dims) # 参数被除了适用于tanh的标准差

    for i in range(0,num_iteration):
        # 前向传播
        ## 是否随机丢弃节点
        if keep_prob == 1:
            AL, cache = reg_utils.forward_propagation(X,parameters)
        elif keep_prob<1:
            AL, cache = reg_utils.forward_propagation_with_dropout(X,parameters,keep_prob=keep_prob)
        else:
            print("keep_prob参数错误！程序退出。")
            exit

        # 计算成本
        ## 是否使用二范数,随机丢弃节点的计算成本没有意义
        if lambd == 0:
            cost = reg_utils.compute_cost(AL,Y)
        else:
            cost = reg_utils.compute_cost_with_regularization(AL,Y,parameters,lambd)

        # 反向传播
        if(lambd == 0 and keep_prob == 1):#不正则化
            grads = reg_utils.backward_propagation(X,Y,cache)
        elif(lambd!=0 and keep_prob ==1):#L2正则化
            grads = reg_utils.backward_propagation_with_regulation(X,Y,cache,lambd)
        elif(lambd == 0 and keep_prob!=1):#随机丢弃节点正则化
            grads = reg_utils.backward_propagation_with_dropout(X,Y,cache,keep_prob)
        else:
            print("没写两个都开的函数...")
            exit

        # 更新参数
        parameters = reg_utils.update_parameters(parameters,grads,learning_rate)

        #记录成本函数
        if i%1000 == 0:
            costs.append(cost)
            if (print_cost and i % 10000 == 0):
                print("第"+str(i)+"次迭代的成本是："+str(cost))

    return costs,parameters