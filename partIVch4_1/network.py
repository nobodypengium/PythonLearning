from keras.models import Sequential  # 该模块允许多个网络层线性堆叠，也允许在已有的网络上加一些层
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate  # concatenate函数用来从axis维度开始连接两个矩阵
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate  # 连接层
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras import backend as K

from IPython.display import SVG  # 画图
from keras.utils.vis_utils import model_to_dot  # 画图
from keras.utils import plot_model  # 画图

K.set_image_data_format('channels_first')

import time  # 时间操作转换
import cv2  # 图像处理
import os
import numpy as np
import sys
from numpy import genfromtxt
import pandas as pd  # 数据结构 数据处理
import tensorflow as tf
import partIVch4_1.fr_utils as fr_utils  # 导入数据定义filter大小
from partIVch4_1.inception_blocks_v2 import *  # 定义网络模块

np.set_printoptions(threshold=sys.maxsize)  # 输出数组的时候完全输出，不需要省略号将中间数据省略


def triplet_loss(y_true, y_pred, alpha=0.2):
    """
    L(A,P,N)=max(‖f(A)-f(P)‖^2-‖f(A)-f(N)‖^2+α,0)
    :param y_true:
    :param y_pred:列表:anchor, positive, negative (?,128)
    :param alpha:阈值，控制anchor与positive距离和anchor与negative距离的插值
    :return:
    loss:损失值
    """
    # 获取anchor, positive, negative图像编码
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]

    # ‖f(A)-f(P)‖^2
    pos_dict = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)  # 128那个维度加起来，完事是(?,1)

    # ‖f(A)-f(N)‖^2
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)

    # ‖f(A)-f(P)‖^2-‖f(A)-f(N)‖^2+α
    basic_loss = tf.add(tf.subtract(pos_dict, neg_dist), alpha)

    # max(‖f(A) - f(P)‖ ^ 2 -‖f(A) - f(N)‖ ^ 2 + α, 0)并对所有样本求和，loss就是一个数了
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0))

    return loss


def verify(image_path, identity, database, model):
    """
    验证image_path与声称的identity是否是一个人
    :param image_path: 输入图像
    :param identity: 声称的身份
    :param database: 成员名字和对应的编码(1,128)
    :param model: 模型实例
    :return:
    dist:差距
    is_open_door:是否应该开门
    """
    # 计算输入图像编码
    encoding = fr_utils.img_to_encoding(image_path, model)

    # 计算与数据库中保存编码的差距
    dist = np.linalg.norm(encoding - database[identity])

    # 是否开门
    if dist < 0.7:
        print("开门" + str(identity))
        is_door_open = True
    else:
        print("假的" + str(identity))
        is_door_open = False

    return dist, is_door_open

def who_is_it(image_path,database,model):
    """
    1. 计算输入图像的(1,128)编码
    2. 从数据库中找出与目标编码具有最小差距的编码
    :param image_path: 输入图像
    :param database: 数据库
    :param model: 网络模型
    :return:
    min_dist:最小距离
    identity:拥有最小距离的人的身份
    """

    #1.计算输入图像编码
    encoding = fr_utils.img_to_encoding(image_path,model)

    #2.找到最相近的编码
    min_dist = 100 #设置min_dist为足够大的数字
