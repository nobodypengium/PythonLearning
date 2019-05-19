import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from tensorflow.python.framework import ops
import partIVch1.cnn_utils as cnn_utils


def create_placehoders(n_H0, n_W0, n_C0, n_y):
    """
    创建占位符，TF要求对输入变量创建占位符，并告诉它你要输入多大的变量NONE表示可变数量
    :param n_H0:
    :param n_W0:
    :param n_C0:
    :param n_y:
    :return: 占位符X和Y
    """

    X = tf.placeholder(tf.float32, [None, n_H0, n_W0, n_C0])  # 不知道会有多少个数据输入，但是每个样本的宽高和通道数一样
    Y = tf.placeholder(tf.float32, [None, n_y])

    return X, Y


def initialize_parameters():
    """
    初始化权值矩阵
    :return:
    """
    tf.set_random_seed(1)

    W1 = tf.get_variable("W1", [4, 4, 3, 8], initializer=tf.contrib.layers.xavier_initializer(seed=0))  # 第一层有8个filter
    W2 = tf.get_variable("W2", [2, 2, 8, 16], initializer=tf.contrib.layers.xavier_initializer(seed=0))

    parameters = {
        "W1":W1,
        "W2":W2
    }

    return parameters

# def forward_propagation(X,parameters):
#     """
#     前向传播
#     CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FULLCONNECTED
#     :param X:
#     :param parameters:
#     :return:
#     """
#
#     W1 = parameters["W1"]
#     W2 = parameters["W2"]
#
#     #CONV2D
#     Z1 = tf.nn.conv2d(X,W1,[1,1,1,1],padding="SAME")
#
#     #ReLU
#     A1 = tf.nn.relu(Z1)
#
#     #MAXPOOL
#     P1 = tf.nn.max_pool(A1,ksize=[1,8,8,1],strides=[1,8,8,1],padding="SAME")
#
#
#     #CONV2D
#     Z2 = tf.nn.conv2d(P1,W2,[1,1,1,1],padding="SAME")
#
#     #RELU
#     A2 = tf.nn.relu(Z2)
#
#     #MAXPOOL
#     P2 = tf.nn.max_pool(A2,ksize=[1,4,4,1],strides=[1,4,4,1],padding="SAME")
#
#     #一维化
#     P = tf.contrib.layers.flatten(P2)
#
#     #全连接层
#     Z3 = tf.contrib.layers.fully_connected(P,6,activation_fn=None)
#
#     return Z3

def forward_propagation(X,parameters):
    """
    实现前向传播
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED

    参数：
        X - 输入数据的placeholder，维度为(输入节点数量，样本数量)
        parameters - 包含了“W1”和“W2”的python字典。

    返回：
        Z3 - 最后一个LINEAR节点的输出

    """
    W1 = parameters['W1']
    W2 = parameters['W2']

    #Conv2d : 步伐：1，填充方式：“SAME”
    Z1 = tf.nn.conv2d(X,W1,strides=[1,1,1,1],padding="SAME")
    #ReLU ：
    A1 = tf.nn.relu(Z1)
    #Max pool : 窗口大小：8x8，步伐：8x8，填充方式：“SAME”
    P1 = tf.nn.max_pool(A1,ksize=[1,8,8,1],strides=[1,8,8,1],padding="SAME")

    #Conv2d : 步伐：1，填充方式：“SAME”
    Z2 = tf.nn.conv2d(P1,W2,strides=[1,1,1,1],padding="SAME")
    #ReLU ：
    A2 = tf.nn.relu(Z2)
    #Max pool : 过滤器大小：4x4，步伐：4x4，填充方式：“SAME”
    P2 = tf.nn.max_pool(A2,ksize=[1,4,4,1],strides=[1,4,4,1],padding="SAME")

    #一维化上一层的输出
    P = tf.contrib.layers.flatten(P2)

    #全连接层（FC）：使用没有非线性激活函数的全连接层
    Z3 = tf.contrib.layers.fully_connected(P,6,activation_fn=None)

    return Z3

def compute_cost(Z3,Y):
    """
    计算softmax层并计算成本，这两个函数在TF里合二为一了
    :param Z3:
    :param Y:
    :return:
    """
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z3,labels=Y))

    return cost

# def model(X_train,Y_train,X_test,Y_test,learning_rate=0.009,num_epoches=100,minibatch_size=64,print_cost=True,isPlot=True):
#     """
#     整个模型，使用小批量下降
#     :param X_train:
#     :param Y_train:
#     :param X_test:
#     :param Y_test:
#     :param learning_rate:
#     :param num_epoches:
#     :param minibatch_size:
#     :param print_cost:
#     :param isPlot:
#     :return:
#     """
#
#     ops.reset_default_graph()
#     tf.set_random_seed(1)
#     seed = 3
#     (m,n_H0,n_W0,n_C0) = X_train.shape
#     n_y = Y.train.shape[1]
#     costs = []
#
#     #创建占位符
#     X, Y = create_placehoders(n_H0,n_W0,n_C0,n_y)
#
#     #初始化参数
#     parameters = initialize_parameters()
#
#     #前向传播
#     Z3 = forward_propagation(X,parameters)
#
#     #Sigmoid 和 计算成本
#     cost = compute_cost(Z3,Y)
#
#     #选择反向传播的优化器
#     optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
#
#     #初始化所有变量
#     init = tf.global_variables_initializer()
#
#     with tf.Session() as sess:
#         #初始化所有变量
#         sess.run(init)
#
#         for epoch in range(num_epoches):
#             minibatch_cost = 0
#             num_minibatches = int(m/minibatch_size)
#             seed = seed+1
#             minibatches = cnn_utils.random_mini_batches(X_train,Y_train,minibatch_size,seed)
#
#             for minibatch in minibatches:
#                 #数据块
#                 (minibatch_X,minibatch_Y) = minibatch
#                 #最小化成本
#                 _,temp_cost = sess.run([optimizer,cost],feed_dict={X:minibatch_X, Y:minibatch_Y})
#                 #累加成本值
#                 minibatch_cost += temp_cost/num_minibatches
#
#             if print_cost:
#                 if epoch % 5 ==0
#                     print("第" + str(epoch) + "代，成本值为：" + str(minibatch_cost))
#
#             if epoch % 1 == 0:
#                 costs.append(minibatch_cost)
#
#         #绘制成本曲线
#         if isPlot:
#             plt.plot(np.squeeze(costs))
#             plt.ylabel('cost')
#             plt.xlabel('iterations (per tens)')
#             plt.title('Learning rate = ' + str(learning_rate))
#
#         #预测
#         predict_op = tf.arg_max(Z3,1)
#         corrent_prediction = tf.equal(predict_op, tf.arg_max(Y,1))
#
#         ##计算准确度
#         accuracy = tf.reduce_mean(tf.cast(corrent_prediction, "float"))
#         print("corrent_prediction accuracy= " + str(accuracy))
#
#         train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
#         test_accuary = accuracy.eval({X: X_test, Y: Y_test})
#
#         print("训练集准确度：" + str(train_accuracy))
#         print("测试集准确度：" + str(test_accuary))
#
#         return (train_accuracy, test_accuary, parameters)

def model(X_train, Y_train, X_test, Y_test, learning_rate=0.009,
         num_epochs=100,minibatch_size=64,print_cost=True,isPlot=True):
    """
    使用TensorFlow实现三层的卷积神经网络
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED

    参数：
        X_train - 训练数据，维度为(None, 64, 64, 3)
        Y_train - 训练数据对应的标签，维度为(None, n_y = 6)
        X_test - 测试数据，维度为(None, 64, 64, 3)
        Y_test - 训练数据对应的标签，维度为(None, n_y = 6)
        learning_rate - 学习率
        num_epochs - 遍历整个数据集的次数
        minibatch_size - 每个小批量数据块的大小
        print_cost - 是否打印成本值，每遍历100次整个数据集打印一次
        isPlot - 是否绘制图谱

    返回：
        train_accuracy - 实数，训练集的准确度
        test_accuracy - 实数，测试集的准确度
        parameters - 学习后的参数
    """
    ops.reset_default_graph()  #能够重新运行模型而不覆盖tf变量
    tf.set_random_seed(1)    #确保你的数据和我一样
    seed = 3                 #指定numpy的随机种子
    (m , n_H0, n_W0, n_C0) = X_train.shape
    n_y = Y_train.shape[1]
    costs = []

    #为当前维度创建占位符
    X , Y = create_placehoders(n_H0, n_W0, n_C0, n_y)

    #初始化参数
    parameters = initialize_parameters()

    #前向传播
    Z3 = forward_propagation(X,parameters)

    #计算成本
    cost = compute_cost(Z3,Y)

    #反向传播，由于框架已经实现了反向传播，我们只需要选择一个优化器就行了
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    #全局初始化所有变量
    init = tf.global_variables_initializer()

    #开始运行
    with tf.Session() as sess:
        #初始化参数
        sess.run(init)
        #开始遍历数据集
        for epoch in range(num_epochs):
            minibatch_cost = 0
            num_minibatches = int(m / minibatch_size) #获取数据块的数量
            seed = seed + 1
            minibatches = cnn_utils.random_mini_batches(X_train,Y_train,minibatch_size,seed)

            #对每个数据块进行处理
            for minibatch in minibatches:
                #选择一个数据块
                (minibatch_X,minibatch_Y) = minibatch
                #最小化这个数据块的成本
                _ , temp_cost = sess.run([optimizer,cost],feed_dict={X:minibatch_X, Y:minibatch_Y})

                #累加数据块的成本值
                minibatch_cost += temp_cost / num_minibatches

            #是否打印成本
            if print_cost:
                #每5代打印一次
                if epoch % 5 == 0:
                    print("当前是第 " + str(epoch) + " 代，成本值为：" + str(minibatch_cost))

            #记录成本
            if epoch % 1 == 0:
                costs.append(minibatch_cost)

        #数据处理完毕，绘制成本曲线
        if isPlot:
            plt.plot(np.squeeze(costs))
            plt.ylabel('cost')
            plt.xlabel('iterations (per tens)')
            plt.title("Learning rate =" + str(learning_rate))
            plt.show()

        #开始预测数据
        ## 计算当前的预测情况
        predict_op = tf.arg_max(Z3,1)
        corrent_prediction = tf.equal(predict_op , tf.arg_max(Y,1))

        ##计算准确度
        accuracy = tf.reduce_mean(tf.cast(corrent_prediction,"float"))
        print("corrent_prediction accuracy= " + str(accuracy))

        train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
        test_accuary = accuracy.eval({X: X_test, Y: Y_test})

        print("训练集准确度：" + str(train_accuracy))
        print("测试集准确度：" + str(test_accuary))

        return (train_accuracy,test_accuary,parameters)