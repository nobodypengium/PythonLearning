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
    a = np.zeros([n_a, m, T_x])
    y_pred = np.zeros([n_y, m, T_x])

    # 初始化a_next，为输入第一个时间步的激活值
    a_next = a0

    # 遍历所有时间步
    for t in range(T_x):
        # 更新隐藏激活值a_next,预测值y_pred与cache
        a_next, yt_pred, cache = rnn_cell_forward(x[:, :, t], a_next, parameters)

        # 保存a_next
        a[:, :, t] = a_next  # a0并不存在这里面

        # 保存预测值
        y_pred[:, :, t] = yt_pred

        # 保存cache
        caches.append(cache)

    # 保存反向传播所需参数
    caches = (caches, x)

    return a, y_pred, caches


def lstm_cell_forward(xt, a_prev, c_prev, parameters):
    """
    构建单个LSTM单元
    :param xt: 当前时间步的输入
    :param a_prev: 前一个时间步的激活值
    :param c_prev: 前一个时间步的记忆细胞值
    :param parameters: W和b的参数
    :return: a_next:下一个隐藏状态 c_next:下一个记忆状态 yt_pred:在t时间的预测 cache:反向传播需要的参数包括a_next,c_next,a_prev,c_prev,x_t,parameters
    """

    # 取出参数
    Wf = parameters["Wf"]
    bf = parameters["bf"]
    Wi = parameters["Wi"]
    bi = parameters["bi"]
    Wc = parameters["Wc"]
    bc = parameters["bc"]
    Wo = parameters["Wo"]
    bo = parameters["bo"]
    Wy = parameters["Wy"]
    by = parameters["by"]

    # 获取维度信息
    n_x, m = xt.shape
    n_y, n_a = Wy.shape

    # 连接a_prev和x_t
    contact = np.zeros([n_a + n_x, m])
    contact[:n_a, :] = a_prev
    contact[n_a:, :] = xt

    # 计算遗忘门ft更新门it更新单元cct输出门ot
    ft = rnn_utils.sigmoid(np.dot(Wf, contact) + bf)
    it = rnn_utils.sigmoid(np.dot(Wi, contact) + bi)
    cct = np.tanh(np.dot(Wc, contact) + bc)
    c_next = ft * c_prev + it * cct
    ot = rnn_utils.sigmoid(np.dot(Wo, contact) + bo)
    a_next = ot * np.tanh(c_next)
    yt_pred = rnn_utils.softmax(np.dot(Wy, a_next) + by)
    cache = (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters)

    return a_next, c_next, yt_pred, cache


def lstm_forward(x, a0, parameters):
    """
    把LSTM模块连接起来，构成LSTM网络
    :param x: 所有时间步的输入 (n_x,m,T_x)
    :param a0: 初始化隐藏状态 (n_a,m)
    :param parameters:所有参数Wb
    :return:a:所有时间步的激活值 y:所有时间步的预测值 caches:反向传播所需信息
    """

    # 初始化caches
    caches = []

    # 获取xt与Wy维度信息，从最少的变量中获取到全部维度信息
    n_x, m, T_x = x.shape
    n_y, n_a = parameters["Wy"].shape

    #用0初始化所有输出值，根据上面获取到的维度，先填0
    a = np.zeros([n_a,m,T_x])
    c = np.zeros([n_a,m,T_x])
    y = np.zeros([n_y,m,T_x])

    #初始化a_next和c_next
    a_next = a0
    c_next = np.zeros([n_a,m])

    #一个一个时间步的网络走
    for t in range(T_x):
        #用刚才写的模块更新隐藏值、记忆状态、预测值和cache
        a_next,c_next,yt_pred,cache = lstm_cell_forward(x[:,:,t],a_next,c_next,parameters)

        #保存各种变量，这时候初始化的作用就体现出来了
        a[:,:,t]=a_next
        y[:,:,t]=yt_pred
        c[:,:,t]=c_next
        caches.append(cache)

    #保存反向传播所需参数
    caches = (caches,x)

    return a,y,c,caches

def rnn_cell_backward(da_next,cache):
    """
    实现反向传播，rnn经常用到的tanh(x)的倒数是1-tanh^2
    反向传播过程：计算d参数>通过d参数更新参数
    用到了哪些变量，只有a吗?
    :param da_next: 下一隐藏状态的损失函数
    :param cache: 前向传播的输出，用于反向传播
    :return:梯度
    """
    # 从cache中读取反向传播需要的值，注意这里是单步的
    a_next,a_prev,xt,parameters = cache

    #从parameters中获取参数
    Wax = parameters["Wax"]
    Waa = parameters["Waa"]
    Wya = parameters["Wya"]
    ba = parameters["ba"]
    by = parameters["by"]

    #dtanh(u)/du = (1-tanh(u)^2)du u=da_next 其中 a_next = Waxx<t>+Waaa<t-1>+b
    dtanh = (1-np.square(a_next)) * da_next

    #Wax的梯度
    dxt = np.dot(Wax.T,dtanh)
    dWax = np.dot(dtanh,xt.T)

    #Waa的梯度
    da_prev = np.dot(Waa.T,dtanh)
    dWaa = np.dot(dtanh,a_prev.T)

    #b的梯度
    dba = np.sum(dtanh, keepdims=True, axis=1)

    #保存提u自定
    gradients = {"dxt": dxt, "da_prev": da_prev, "dWax": dWax, "dWaa": dWaa, "dba": dba}
    return gradients
