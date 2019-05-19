import numpy as np
import h5py
import matplotlib.pyplot as plt

def zero_pad(X,pad):

    #样本，高度，宽度，通道数填充0
    X_paded = np.pad(X,((0,0),(pad,pad),(pad,pad),(0,0)),'constant',constant_values=(0,0))
    return X_paded

def conv_single_step(a_slice_prev,W,b):
    """
    进行一步的卷积，
    :param a_slice_prev:(n_H^([l]),n_W^([l]),n_c^([l]))
    :param W:(f^([l]),f^([l]),n_c^[l-1] )
    :param b:(1,1,1)
    :return:
    """

    # s = np.multiply(a_slice_prev,W)+b
    s = a_slice_prev*W+b
    Z = np.sum(s)
    return Z

def conv_forward(A_prev, W, b, hparameters):

    #前层样本数，图像高度、宽度、前层滤波器数量（通道数）
    (m,n_H_prev,n_W_prev,n_C_prev) = A_prev.shape

    #权重 高度 宽度 每个多少层 多少个
    (f,f,n_C_prev,n_C) = W.shape

    #超参数步长和填充数量
    stride = hparameters["stride"]
    pad  = hparameters["pad"]

    #卷积后图像宽高
    n_H = int((n_H_prev+2*pad-f)/stride)+1
    n_W = int((n_W_prev+2*pad-f)/stride)+1

    #0初始化卷积输出
    Z = np.zeros((m,n_H,n_W,n_C))

    #填充
    A_prev_pad = zero_pad(A_prev,pad)

    for i in range(m):
        a_prev_pad = A_prev_pad[i]
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f

                    a_slice_prev = a_prev_pad[vert_start:vert_end,horiz_start:horiz_end,:]
                    #卷积之后对于某个样本，某一通道(c)上，某个位置(h,w)的值为抠出来的一小个立方体阵与作为权重的立方体阵的卷积
                    Z[i,h,w,c] = conv_single_step(a_slice_prev,W[:,:,:,c],b[0,0,0,c])


    cache = (A_prev,W,b,hparameters)

    return (Z,cache)

def pool_forward(A_prev,hparameters,mode="max"):
    """
    池化层前向传播
    :param A_prev: 输入数据 m,n_H_prev,n_W_prev,n_C_prev
    :param hparameters: stride,f
    :param mode: 平均值池化or最大值池化
    :return: 池化后立方体阵，输入与超参数字典
    """

    #输入数据的基本信息
    (m,n_H_prev,n_W_prev,n_C_prev) = A_prev.shape

    #超参数
    f = hparameters["f"]
    stride = hparameters["stride"]

    #输出维度
    n_H = int((n_H_prev - f)/stride)+1
    n_W = int((n_W_prev - f)/stride)+1
    n_C = n_C_prev #池化之后通道数不变

    #初始化输出矩阵
    A = np.zeros((m,n_H,n_W,n_C))

    for i in  range(m):
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f

                    a_slice_prev = A_prev[i,vert_start:vert_end,horiz_start:horiz_end,c]

                    if mode == "max":
                        A[i,h,w,c] = np.max(a_slice_prev)
                    elif mode == "average":
                        A[i,h,w,c] = np.mean(a_slice_prev)

    cache = (A_prev,hparameters)
    return A,cache

