import numpy as np
from partVch1 import rnn_utils


def rnn_cell_forward(xt, a_prev, parameters):
    """
    实现一个RNN单元的前向传播
    :param xt: 时间步t时输入的数据 (n_x,m)
    :param a_prev:  时间步t-1时输入的数据 (n_a,m)
    :param parameters: 参数，Wax,Waa,Wya,ba,by
    :return:
    a_next:下一个隐藏状态 (n_a,m)
    yt_pred:时间步t时的预测
    cache:反向传播需要的a_next,a_prev,xt,parameters
    """
    # 从parameter读取参数
    Wax = parameters["Wax"]
    Waa = parameters["Waa"]
    Wya = parameters["Wya"]
    ba = parameters["ba"]
    by = parameters["by"]

    # 计算下一个激活值
    a_next = np.tanh(np.dot(Waa, a_prev) + np.dot(Wax, xt) + ba)

    # 计算当前单元输出
    yt_pred = rnn_utils.softmax(np.dot(Wya, a_next) + by)

    # 保存反向传播需要的值
    cache = (a_next, a_prev, xt, parameters)

    return a_next, yt_pred, cache


def rnn_forward(x, a0, parameters):
    """
    实现多个时间步的神经网络的前向传播
    1.创建零向量a，保存RNN中所有隐藏状态
    2.循环所有时间步
    :param x: 输入时间序列 (n_x每个样本的长度,m个样本,T_x个时间步)
    :param a0: 隐藏状态，初始化为全0
    :param parameters: W和b
    :return:
    a:所有时间步的隐藏状态 (n_a,m,T_x)
    y_pred:所有时间步的预测 (n_y,m,T_y)
    caches:反向传播所需的cache(包含Wb，各层的输入输出，x)
    """

    # 初始化caches，收集所有时间步的cache
    caches = []

    # 获取x, Wya的维度信息，该信息用来初始化输出
    n_x, m, T_x = x.shape
    n_y, n_a = parameters["Wya"].shape

    # 初始化a和y为0
    a = np.zeros(n_a, m, T_x)
    y_pred = np.zeros([n_y, m, T_x])

    # 初始化a_next，为输入第一个时间步的激活值
    a_next = a0

    # 遍历所有时间步
    for t in range(T_x):
        # 更新隐藏激活值a_next,预测值y_pred与cache
        a_next, yt_pred, cache = rnn_cell_forward(x[:, :, t], a_next, parameters)

        # 保存a_next
        a[:, :, t] = a_next

        # 保存预测值
        y_pred[:, :, t] = yt_pred

        # 保存cache
        caches.append(cache)
    
    #保存反向传播所需参数
    caches = (caches,x)

    return a,y_pred,caches