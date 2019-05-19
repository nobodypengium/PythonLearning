import numpy as np
import h5py
import matplotlib.pyplot as plt


def zero_pad(X, pad):
    # 样本，高度，宽度，通道数填充0
    X_paded = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant', constant_values=(0, 0))
    return X_paded


def conv_single_step(a_slice_prev, W, b):
    """
    进行一步的卷积，
    :param a_slice_prev:(n_H^([l]),n_W^([l]),n_c^([l]))
    :param W:(f^([l]),f^([l]),n_c^[l-1] )
    :param b:(1,1,1)
    :return:
    """

    # s = np.multiply(a_slice_prev,W)+b
    s = a_slice_prev * W + b
    Z = np.sum(s)
    return Z


def conv_forward(A_prev, W, b, hparameters):
    # 前层样本数，图像高度、宽度、前层滤波器数量（通道数）
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

    # 权重 高度 宽度 每个多少层 多少个
    (f, f, n_C_prev, n_C) = W.shape

    # 超参数步长和填充数量
    stride = hparameters["stride"]
    pad = hparameters["pad"]

    # 卷积后图像宽高
    n_H = int((n_H_prev + 2 * pad - f) / stride) + 1
    n_W = int((n_W_prev + 2 * pad - f) / stride) + 1

    # 0初始化卷积输出
    Z = np.zeros((m, n_H, n_W, n_C))

    # 填充
    A_prev_pad = zero_pad(A_prev, pad)

    for i in range(m):
        a_prev_pad = A_prev_pad[i]
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f

                    a_slice_prev = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]
                    # 卷积之后对于某个样本，某一通道(c)上，某个位置(h,w)的值为抠出来的一小个立方体阵与作为权重的立方体阵的卷积
                    Z[i, h, w, c] = conv_single_step(a_slice_prev, W[:, :, :, c], b[0, 0, 0, c])

    cache = (A_prev, W, b, hparameters)

    return (Z, cache)


def pool_forward(A_prev, hparameters, mode="max"):
    """
    池化层前向传播
    :param A_prev: 输入数据 m,n_H_prev,n_W_prev,n_C_prev
    :param hparameters: stride,f
    :param mode: 平均值池化or最大值池化
    :return: 池化后立方体阵，输入与超参数字典
    """

    # 输入数据的基本信息
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

    # 超参数
    f = hparameters["f"]
    stride = hparameters["stride"]

    # 输出维度
    n_H = int((n_H_prev - f) / stride) + 1
    n_W = int((n_W_prev - f) / stride) + 1
    n_C = n_C_prev  # 池化之后通道数不变

    # 初始化输出矩阵
    A = np.zeros((m, n_H, n_W, n_C))

    for i in range(m):
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f

                    a_slice_prev = A_prev[i, vert_start:vert_end, horiz_start:horiz_end, c]

                    if mode == "max":
                        A[i, h, w, c] = np.max(a_slice_prev)
                    elif mode == "average":
                        A[i, h, w, c] = np.mean(a_slice_prev)

    cache = (A_prev, hparameters)
    return A, cache


def conv_backward(dZ, cache):
    """
    卷积层反向传播
    :param dZ:
    :param cache:
    :return:
    """

    # 反向传播所需要的参数
    (A_prev, W, b, hparameters) = cache
    pad = hparameters["pad"]
    stride = hparameters["stride"]

    # 控制循环所需参数
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (m, n_H, n_W, n_C) = dZ.shape
    (f, f, n_C_prev, n_C) = W.shape

    # 用0初始化各个梯度大小
    dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))
    dW = np.zeros((f, f, n_C_prev, n_C))
    db = np.zeros((1, 1, 1, n_C))

    # 前向传播使用pad反向传播也要使用因为保证数据结构一致
    A_prev_pad = zero_pad(A_prev, pad)
    dA_prev_pad = zero_pad(dA_prev, pad)

    for i in range(m):
        a_prev_pad = A_prev_pad[i]
        da_prev_pad = dA_prev_pad[i]

        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    # 定位切片
                    # 应该*stride吧
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f

                    a_slice = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]

                    # 反向传播三个计算公式，再w和h上进行累加
                    da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += W[:, :, :, c] * dZ[i, h, w, c]
                    dW[:, :, :, c] += a_slice * dZ[i, h, w, c]
                    db[:, :, :, c] += dZ[i, h, w, c]

        dA_prev[i, :, :, :] = da_prev_pad[pad:-pad, pad:-pad, :]  # 取出某个样本的dA的所有通道

    return (dA_prev, dW, db)


def create_mask_from_window(x):
    mask = x == np.max(x)
    return mask


def distribute_value(dz, shape):
    """
    将dZ平均分配给所有卷积层的位置，因为它们对输出的作用一样
    :param dz:
    :param shape:
    :return:
    """

    (n_H, n_W) = shape
    average = dz / (n_H * n_W)

    a = np.ones(shape) * average

    return a

def pool_backward(dA,cache,mode = "max"):
    (A_prev, hparameters) = cache

    f = hparameters["f"]
    stride = hparameters["stride"]

    (m,n_H_prev,n_W_prev,n_C_prev) = A_prev.shape
    (m,n_H,n_W,n_C) = dA.shape

    #初始化输出结构
    dA_prev = np.zeros_like(A_prev)

    for i in range(m):
        a_prev = A_prev[i]
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    vert_start = h*stride
                    vert_end = vert_start + f
                    horiz_start = w*stride
                    horiz_end = horiz_start + f

                    if mode == "max":
                        a_prev_slice = a_prev[vert_start:vert_end,horiz_start:horiz_end,c]
                        mask = create_mask_from_window(a_prev_slice)
                        dA_prev[i,vert_start:vert_end,horiz_start:horiz_end,c] += np.multiply(mask,dA[i,h,w,c])#可能第1、2个参数有重叠，所以+=，权重全部来自于最大值的哪一项
                    elif mode == "average":
                        da = dA[i,h,w,c]
                        shape = (f,f)
                        dA_prev[i,vert_start:vert_end,horiz_start:horiz_end,c] += distribute_value(da,shape)
    return dA_prev