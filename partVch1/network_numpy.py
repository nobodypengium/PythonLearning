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

    # 用0初始化所有输出值，根据上面获取到的维度，先填0
    a = np.zeros([n_a, m, T_x])
    c = np.zeros([n_a, m, T_x])
    y = np.zeros([n_y, m, T_x])

    # 初始化a_next和c_next
    a_next = a0
    c_next = np.zeros([n_a, m])

    # 一个一个时间步的网络走
    for t in range(T_x):
        # 用刚才写的模块更新隐藏值、记忆状态、预测值和cache
        a_next, c_next, yt_pred, cache = lstm_cell_forward(x[:, :, t], a_next, c_next, parameters)

        # 保存各种变量，这时候初始化的作用就体现出来了
        a[:, :, t] = a_next
        y[:, :, t] = yt_pred
        c[:, :, t] = c_next
        caches.append(cache)

    # 保存反向传播所需参数
    caches = (caches, x)

    return a, y, c, caches


def rnn_cell_backward(da_next, cache):
    """
    实现反向传播，rnn经常用到的tanh(x)的倒数是1-tanh^2
    反向传播过程：计算d参数>通过d参数更新参数
    用到了哪些变量，只有a吗?
    :param da_next: 下一隐藏状态的损失函数
    :param cache: 前向传播的输出，用于反向传播
    :return:梯度
    """
    # 从cache中读取反向传播需要的值，注意这里是单步的
    a_next, a_prev, xt, parameters = cache

    # 从parameters中获取参数
    Wax = parameters["Wax"]
    Waa = parameters["Waa"]
    Wya = parameters["Wya"]
    ba = parameters["ba"]
    by = parameters["by"]

    # dtanh(u)/du = (1-tanh(u)^2)du u=da_next 其中 a_next = Waxx<t>+Waaa<t-1>+b
    dtanh = (1 - np.square(a_next)) * da_next

    # Wax的梯度
    dxt = np.dot(Wax.T, dtanh)
    dWax = np.dot(dtanh, xt.T)

    # Waa的梯度
    da_prev = np.dot(Waa.T, dtanh)
    dWaa = np.dot(dtanh, a_prev.T)

    # b的梯度
    dba = np.sum(dtanh, keepdims=True, axis=1)

    # 保存提u自定
    gradients = {"dxt": dxt, "da_prev": da_prev, "dWax": dWax, "dWaa": dWaa, "dba": dba}
    return gradients


def rnn_backward(da, caches):
    """
    在整个输入序列上实现RNN反向传播
    :param da: 所有隐藏状态的梯度
    :param caches: 包含前向传播路给反向传播的信息
    :return: gradients:
    dx - 输入数据的梯度 (n_x,m,T_x)
    da0 - 初始化隐藏状态的梯度 (n_a,m)
    dWax - 权重的梯度 (n_a,n_x)
    dWaa - 隐藏状态权值的梯度 (n_a, n_a)
    dba - 偏置的梯度 (n_a,1)
    """
    # 获取时间步t=1时候的值
    caches, x = caches
    a1, a0, x1, parameters = caches[0]

    # 获取维度信息
    n_a, m, T_x = da.shape
    n_x, m = x1.shape

    # 初始化梯度
    dx = np.zeros([n_x, m, T_x])
    dWax = np.zeros([n_a, n_x])
    dWaa = np.zeros([n_a, n_a])
    dba = np.zeros([n_a, 1])
    da0 = np.zeros([n_a, m])
    da_prevt = np.zeros([n_a, m])

    # 处理所有时间步
    for t in reversed(range(T_x)):
        # 计算时间步t时的梯度
        gradients = rnn_cell_backward(da[:, :, t] + da_prevt, caches[t])

        # 获取梯度
        dxt, da_prevt, dWaxt, dWaat, dbat = gradients["dxt"], gradients["da_prev"], gradients["dWax"], gradients[
            "dWaa"], gradients["dba"]

        # 累加到全局，因为在RNN中所有时间步共享一套参数
        dx[:, :, t] = dxt
        dWax += dWaxt
        dWaa += dWaat
        dba += dbat

    # 通过所有时间步反向传播过的a的梯度
    da0 = da_prevt

    # 保存所有参数到字典
    gradients = {"dx": dx, "da0": da0, "dWax": dWax, "dWaa": dWaa, "dba": dba}

    return gradients


def lstm_cell_backward(da_next, dc_next, cache):
    """
    LSTM单个时间步的反向传播
    :param da_next: 下一隐藏状态的梯度
    :param dc_next: 下一记忆单元的梯度
    :param cache: 前向传播对反向传播有用的信息
    :return: 梯度字典，包含变量梯度与参数梯度
    """
    #获取来自前向传播的信息
    (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters) = cache

    #获取维度信息，以便创建用于存储的零向量
    n_x, m = xt.shape
    n_a, m = a_next.shape

    #计算门的梯度
    dot = da_next * np.tanh(c_next) * ot * (1 - ot)
    dcct = (dc_next * it + ot * (1 - np.square(np.tanh(c_next))) * it * da_next) * (1 - np.square(cct))
    dit = (dc_next * cct + ot * (1 - np.square(np.tanh(c_next))) * cct * da_next) * it * (1 - it)
    dft = (dc_next * c_prev + ot * (1 - np.square(np.tanh(c_next))) * c_prev * da_next) * ft * (1 - ft)

    #计算参数的导数
    concat = np.concatenate((a_prev, xt), axis=0).T
    dWf = np.dot(dft, concat)
    dWi = np.dot(dit, concat)
    dWc = np.dot(dcct, concat)
    dWo = np.dot(dot, concat)
    dbf = np.sum(dft, axis=1, keepdims=True)
    dbi = np.sum(dit, axis=1, keepdims=True)
    dbc = np.sum(dcct, axis=1, keepdims=True)
    dbo = np.sum(dot, axis=1, keepdims=True)

    #计算先前隐藏状态、记忆状态、输入导数
    da_prev = np.dot(parameters["Wf"][:, :n_a].T, dft) + np.dot(parameters["Wc"][:, :n_a].T, dcct) + np.dot(
        parameters["Wi"][:, :n_a].T, dit) + np.dot(parameters["Wo"][:, :n_a].T, dot)
    dc_prev = dc_next * ft + ot * (1 - np.square(np.tanh(c_next))) * ft * da_next
    dxt = np.dot(parameters["Wf"][:, n_a:].T, dft) + np.dot(parameters["Wc"][:, n_a:].T, dcct) + np.dot(
        parameters["Wi"][:, n_a:].T, dit) + np.dot(parameters["Wo"][:, n_a:].T, dot)

    #保存梯度到字典
    gradients = {"dxt": dxt, "da_prev": da_prev, "dc_prev": dc_prev, "dWf": dWf, "dbf": dbf, "dWi": dWi, "dbi": dbi,
                 "dWc": dWc, "dbc": dbc, "dWo": dWo, "dbo": dbo}

    return gradients

def lstm_backward(da,caches):
    """
    lstm网络的反向传播
    :param da: 隐藏状态的梯度
    :param caches: 前向传播的信息
    :return: gradients:返回梯度信息
    """

    # #获取t=1的值用来得到维度，得到维度用来初始0梯度
    # caches, x = caches
    # (a1, c1, a0, c0, f1, i1, cc1, o1, x1, parameters) = caches[0]
    #
    # #获取da和x1的维度信息
    # n_a,m,T_x = da.shape
    # n_x,m = x1.shape
    #
    # #初始化梯度
    # dx = np.zeros([n_x,m,T_x])
    # da0 = np.zeros([n_a,m])
    # da_prevt = np.zeros([n_a, m])
    # dc_prevt = np.zeros([n_a, m])
    # dWf = np.zeros([n_a, n_a + n_x])
    # dWi = np.zeros([n_a, n_a + n_x])
    # dWc = np.zeros([n_a, n_a + n_x])
    # dWo = np.zeros([n_a, n_a + n_x])
    # dbf = np.zeros([n_a, 1])
    # dbi = np.zeros([n_a, 1])
    # dbc = np.zeros([n_a, 1])
    # dbo = np.zeros([n_a, 1])
    #
    # # 处理所有时间步
    # for t in reversed(range(T_x)):
    #     # 使用lstm_cell_backward函数计算所有梯度
    #     gradients = lstm_cell_backward(da[:, :, t], dc_prevt, caches[t])
    #     # 保存相关参数
    #     dx[:, :, t] = gradients['dxt']
    #     dWf = dWf + gradients['dWf']
    #     dWi = dWi + gradients['dWi']
    #     dWc = dWc + gradients['dWc']
    #     dWo = dWo + gradients['dWo']
    #     dbf = dbf + gradients['dbf']
    #     dbi = dbi + gradients['dbi']
    #     dbc = dbc + gradients['dbc']
    #     dbo = dbo + gradients['dbo']
    # # 将第一个激活的梯度设置为反向传播的梯度da_prev。
    # da0 = gradients['da_prev']
    #
    # # 保存所有梯度到字典变量内
    # gradients = {"dx": dx, "da0": da0, "dWf": dWf, "dbf": dbf, "dWi": dWi, "dbi": dbi,
    #              "dWc": dWc, "dbc": dbc, "dWo": dWo, "dbo": dbo}
    #
    # return gradients

    # Retrieve values from the first cache (t=1) of caches.
    (caches, x) = caches
    (a1, c1, a0, c0, f1, i1, cc1, o1, x1, parameters) = caches[0]

    ### START CODE HERE ###
    # Retrieve dimensions from da's and x1's shapes (≈2 lines)
    n_a, m, T_x = da.shape
    n_x, m = x1.shape

    # initialize the gradients with the right sizes (≈12 lines)
    dx = np.zeros((n_x, m, T_x))
    da0 = np.zeros((n_a, m))
    da_prevt = np.zeros((n_a, m))
    dc_prevt = np.zeros((n_a, m))
    dWf = np.zeros((n_a, n_a + n_x))
    dWi = np.zeros((n_a, n_a + n_x))
    dWc = np.zeros((n_a, n_a + n_x))
    dWo = np.zeros((n_a, n_a + n_x))
    dbf = np.zeros((n_a, 1))
    dbi = np.zeros((n_a, 1))
    dbc = np.zeros((n_a, 1))
    dbo = np.zeros((n_a, 1))

    # loop back over the whole sequence
    for t in reversed(range(T_x)):
        # Compute all gradients using lstm_cell_backward
        gradients = lstm_cell_backward(da[:, :, t] + da_prevt, dc_prevt, caches[t])
        # Store or add the gradient to the parameters' previous step's gradient
        dx[:, :, t] = gradients['dxt']
        dWf = dWf + gradients['dWf']
        dWi = dWi + gradients['dWi']
        dWc = dWc + gradients['dWc']
        dWo = dWo + gradients['dWo']
        dbf = dbf + gradients['dbf']
        dbi = dbi + gradients['dbi']
        dbc = dbc + gradients['dbc']
        dbo = dbo + gradients['dbo']
    # Set the first activation's gradient to the backpropagated gradient da_prev.
    da0 = gradients['da_prev']

    ### END CODE HERE ###

    # Store the gradients in a python dictionary
    gradients = {"dx": dx, "da0": da0, "dWf": dWf, "dbf": dbf, "dWi": dWi, "dbi": dbi,
                 "dWc": dWc, "dbc": dbc, "dWo": dWo, "dbo": dbo}

    return gradients