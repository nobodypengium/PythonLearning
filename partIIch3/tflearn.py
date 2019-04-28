import numpy as np
import h5py
import matplotlib as plt
import tensorflow as tf
from tensorflow.python.framework import ops
import partIIch3.tf_utils as tf_utils
import time


def linear_function():
    """
    Y=WX+b，该函数的特点是常量+定义的算式
    :return: Y
    """
    np.random.seed(1)

    X = np.random.randn(3, 1)
    W = np.random.randn(4, 3)
    b = np.random.randn(4, 1)

    Y = tf.matmul(W, X) + b

    with tf.Session() as session:
        result = session.run(Y)

    return result


def sigmoid(z):
    x = tf.placeholder(tf.float32, name="x")  # 占位符x
    sigmoid = tf.sigmoid(x)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=1., logits=x)

    with tf.Session() as sess:
        result = sess.run(sigmoid, feed_dict={x: z})  # 用字典给占位符喂数据
        fuck = sess.run(loss, feed_dict={x:z})

        return result, fuck

def one_hot_matrix(labels,C):
    """
    输入一维标签矩阵，返回仅含0，1的独热矩阵
    :param labels: 标签向量
    :param C: 分类个数
    :return: 独热矩阵
    """

    one_hot_matrix = tf.one_hot(indices=labels, depth=C, axis=0) #目录，深度，在纵轴上的x位置加1

    with tf.Session() as sess:
        result = sess.run(one_hot_matrix)

    return result

def ones(shape):
    """
    初始化全1向量
    :param shape: 向量形状
    :return:
    """
    ones = tf.ones(shape)
    with tf.Session() as sess:
        result = sess.run(ones)

    return result