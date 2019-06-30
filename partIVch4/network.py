import time
import os
import sys
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
import partIVch4.nst_utils as nst_utils
import numpy as np
import tensorflow as tf

STYLE_LAYERS

def compute_content_cost(a_C,a_G):
    """
    计算内容代价
    :param a_C: content在选定隐藏层中的激活值
    :param a_G: generate在选定隐藏层中的激活值
    :return: J_content: 代价函数
    """
    #获取a_G维度信息
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    #降维
    a_C_unrolled = tf.transpose(tf.reshape(a_C,[n_H*n_W,n_C]))
    a_G_unrolled = tf.transpose(tf.reshape(a_G,[n_H*n_W,n_C])) #一层矩阵转化为一行矩阵，把宽高降成1维
    #计算内容代价J_content (C,G)=‖a^([l](C))-a^([l][G]) ‖^2
    J_content = (1/(4*n_H*n_W*n_C))*tf.reduce_sum(tf.square(tf.subtract(a_C_unrolled,a_G_unrolled))) #J_content就是一个数了
    return J_content

def gram_matrix(A):
    """
    计算矩阵A的风格矩阵，是图片到某些隐藏层的输出
    :param A: 矩阵 - n_C, n_H*n_W
    :return: GA: A的风格矩阵
    """
    GA = tf.matmul(A, A, transpose_b=True) # 矩阵点乘
    return GA

def compute_layer_style_cost(a_S,a_G):
    """
    计算单隐藏层的风格损失
    1. 获取维度信息
    2. 降维
    3. 计算风格矩阵
    4. 根据风格矩阵计算风格损失
    :param a_S: (1,n_H,n_W,n_C),选定风格图片隐藏层的激活值
    :param a_G: (1,n_H,n_W,n_C),选定生成图片隐藏层的激活值
    :return: J_style_layer:风格损失函数
    """

    #获取维度信息
    m,n_H,n_W,n_C = a_G.get_shape().as_list()

    #降维
    a_S = tf.transpose(tf.reshape(a_S,[n_H*n_W,n_C])) #一行对应一层
    a_G = tf.transpose(tf.reshape(a_G,[n_H*n_W,n_C]))

    #计算风格矩阵
    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)

    #计算风格损失 1/〖(2n_H^([l]) n_w^([l]) n_c^([l]))〗^2  ∑_k▒∑_(k^')▒( 〖G_(kk^')^([l](S))-G_(kk^')^[l](G) )〗^2
    J_style_layer = (1/(4*n_C*n_C*n_H*n_H*n_W*n_W)*tf.reduce_sum(tf.square(tf.subtract(GS,GG))))

    return J_style_layer

def compute_style_cost(model,STYLE_LAYERS)