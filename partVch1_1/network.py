import numpy as np
import random
import time
from partVch1_1 import cllm_utils


def clip(gradients, maxValue):
    """
    使用maxValue修剪梯度
    :param gradients: RNN反向传播产生的梯度字典
    :param maxValue: 阈值
    :return: gradients::修剪后的梯度字典
    """
    dWaa, dWax, dWya, db, dby = gradients['dWaa'], gradients['dWax'], gradients['dWya'], gradients['db'], gradients[
        'dby']

    for gradient in [dWaa, dWax, dWya, db, dby]:
        np.clip(gradient, -maxValue, maxValue, out=gradient)  # 进行修剪再放回原来的变量

    gradients = {"dWaa": dWaa, "dWax": dWax, "dWya": dWya, "db": db, "dby": dby}

    return gradients


def sample(parameters, char_to_ix, seed):
    """
    根据RNN输出的概率分布对字符序列采样
    :param parameters: W和b，这个函数里同时进行了前向传播，因为sample的目的是喂给下一层的
    :param char_to_is: 字符映射到索引的字典
    :param seed: 随机种子
    :return: 列表，长度n，包含采样字符索引
    """
    # 获取参数
    # 从parameters 中获取参数，参数用来获得维度，维度用来初始化输入
    Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']
    vocab_size = by.shape[0]
    n_a = Waa.shape[1]

    # 创建独热编码
    x = np.zeros((vocab_size, 1))
    # 初始化a_prev为0
    a_prev = np.zeros((n_a, 1))
    # 创建空索引列表，生成的字符往里加
    indices = []
    # 检测换行符的flag 初始化为-1
    idx = -1

    # 循环遍历时间步t，并从每个时间步中，根据概率分布抽取一个字符并将其加到indices 如果达到50个字符就停止循环
    counter = 0
    newline_character = char_to_ix["\n"] #找到换行符的索引

    while (idx != newline_character and counter < 50):
        #前向传播，跟抽样合在一起了
        a = np.tanh(np.dot(Wax,x) + np.dot(Waa,a_prev) + b)
        z = np.dot(Wya,a) + by
        y = cllm_utils.softmax(z)

        #设定随机种子
        np.random.seed(counter + seed)

        #从y中取出索引并替换x
        idx = np.random.choice(list(range(vocab_size)),p=y.ravel())
        indices.append(idx)

        #创建独热编码
        x = np.zeros((vocab_size,1))
        x[idx] = 1

        #更新a_prev
        a_prev=a

        #累加
        seed += 1
        counter += 1

    if(counter == 50):
        indices.append(char_to_ix["\n"])

    return indices

def optimize(X,Y,a_prev,parameters,learning_rate=0.01):
    """
    前向传播 -> 反向传播 -> 修剪梯度 ->更新参数 一次优化一个样本
    :param X: 整数列表，映射到词汇表中的【字符】
    :param Y: 整数列表，比x向左移动一个索引，因为前一层的输出y作后一层的输入x
    :param a_prev: 上一层的隐藏状态
    :param parameters: W和b的字典
    :param learning_rate: 学习率
    :return:
    loss:损失
    gradients 参数和变量的梯度
    a[len(X)-1]最后的隐藏状态
    """

    #前向传播
    loss, cache = cllm_utils.rnn_forward(X,Y,a_prev,parameters)

    #反向传播
    gradients, a = cllm_utils.rnn_backward(X,Y,parameters,cache)

    #梯度修剪
    gradients = clip(gradients,5)

    #更新参数
    parameters = cllm_utils.update_parameters(parameters,gradients,learning_rate)

    return loss,gradients,a[len(X)-1]

