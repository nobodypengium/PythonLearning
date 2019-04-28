import numpy as np
import h5py
import matplotlib as plt
import tensorflow as tf
from tensorflow.python.framework import ops
import partIIch3.tf_utils as tf_utils
import time


def create_placeholder(n_x, n_y):
    """
    为输入输出创建占位符，便于在TF会话中导入数据
    :param n_x:每个图片样本的输入维度
    :param n_y:每个图片样本的输出维度
    :return:占位符XY
    """
    X = tf.placeholder(tf.float32, [n_x, None], name="X")
    Y = tf.placeholder(tf.float32, [n_y, None], name="Y")

    return X, Y


def initialize_parameters():
    """
    初始化参数，这里硬编码
    :return:
    """
    tf.set_random_seed(1)

    W1 = tf.get_variable("W1", [25, 12288], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b1 = tf.get_variable("b1", [25, 1], initializer=tf.zeros_initializer())
    W2 = tf.get_variable("W2", [12, 25], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b2 = tf.get_variable("b2", [12, 1], initializer=tf.zeros_initializer())
    W3 = tf.get_variable("W3", [6, 12], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b3 = tf.get_variable("b3", [6, 1], initializer=tf.zeros_initializer())

    parameters = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2,
        "W3": W3,
        "b3": b3}

    return parameters


def forward_propagation(X, parameters):
    """
    用TF前向传播，跟numpy实现的主要区别是最后停在Z3因为用Z3计算损失，另外不需要cache因为TF将算式定义在运算图中
    :param X:
    :param parameters:
    :return:
    """
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]

    Z1 = tf.matmul(W1, X) + b1
    A1 = tf.nn.relu(Z1)
    Z2 = tf.matmul(W2, A1) + b2
    A2 = tf.nn.relu(Z2)
    Z3 = tf.matmul(W3, A2) + b3

    return Z3


def compute_cost(Z3, Y):
    """
    与numpy的实现方式的区别是参数ZL还是AL，因为TF有直接用ZL求cost的函数
    :param Z3:
    :param Y:
    :return:
    """
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)

    # 在指定维度上求均值，跟numpy里sum了再/m很像，这里用到Softmax，最后一层Softmax层共有6个节点，这是通过那6个节点的均值
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

    return cost


def model(X_train, Y_train, X_test, Y_test, learning_rate=0.0001, num_epochs=1500, minibatch_size=32, print_cost=True,
          is_plot=True):
    """
    三层的TF神经网络
    创建占位符->初始化参数->前向传播（构建信息流图）->计算成本->反向传播
    :param X_train:
    :param Y_train:
    :param X_test:
    :param Y_test:
    :param learning_rate:
    :param num_epochs:
    :param minibatch_size:
    :param print_cost:
    :param is_plot:
    :return: 学习后的参数
    """
    ops.reset_default_graph()
    tf.set_random_seed(1)
    seed = 3
    (n_x, m) = X_train.shape
    n_y = Y_train.shape[0]
    costs = []

    # 创建占位符，这个是用于根据字典喂数据的（输入），不需要保存
    X, Y = create_placeholder(n_x, n_y)

    # 初始化参数，这里得到的是变量（输出），因为这部分需要储存
    parameters = initialize_parameters()

    # 前向传播
    Z3 = forward_propagation(X, parameters)

    # 计算成本
    cost = compute_cost(Z3, Y)

    # 反向传播
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # 初始化变量
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        # 初始化
        sess.run(init)

        # 小批量训练的两层循环，外层epoch，内层批量
        for epoch in range(num_epochs):
            epoch_cost = 0
            num_minibatches = int(m / minibatch_size)
            seed = seed + 1
