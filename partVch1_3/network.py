from keras.models import load_model, Model
from keras.layers import Dense, Activation, Dropout, Input, LSTM, Reshape, Lambda, RepeatVector
from keras.initializers import glorot_uniform
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras import backend as K
import numpy as np
import IPython
import sys
from music21 import *
from partVch1_3.grammar import *
from partVch1_3.qa import *
from partVch1_3.preprocess import *
from partVch1_3.music_utils import *
from partVch1_3.data_utils import *
import time

X, Y, n_values, indices_values = load_music_utils()

n_a = 64
reshapor = Reshape((1, 78))
LSTM_cell = LSTM(n_a, return_state=True)  # 定义图层对象，每次都调用这一个图层对象，权重就不会更新，我们需要走完所有Tx图层不更新
densor = Dense(n_values, activation='softmax')  # 第一个参数：该层有几个神经元

# 初始化a0和c0
m = 60
a0 = np.zeros((m, n_a))
c0 = np.zeros((m, n_a))


def djmodel(Tx, n_a, n_values):
    """
    实现由LSTM连接的制作音乐的模型
    :param Tx: 时间步数量
    :param n_a: 激活值数量
    :param n_values: 唯一编码数量
    :return: model: keras模型
    """

    # 输入数据的维度,Tx个时间步,每个输入由n_value种可能
    X = Input((Tx, n_values))

    # 定义a0，初始化
    a0 = Input(shape=(n_a,), name="a0")
    c0 = Input(shape=(n_a,), name="c0")
    a = a0
    c = c0

    # 创建空的output保存LSTM所有时间步的输出
    outputs = []

    # 循环算y然后丢到outputs
    for t in range(Tx):
        # 选择第t个时间步对应的X向量
        x = Lambda(lambda x: X[:, t, :])(X)
        # 重构x为一行的向量
        x = reshapor(x)
        # 单步传播
        a, _, c = LSTM_cell(x, initial_state=[a, c])
        # densor输出并添加到outputs
        out = densor(a)
        outputs.append(out)

    # 创建模型，运用上面的计算，指定输入输出即可
    model = Model(inputs=[X, a0, c0], outputs=outputs)
    return model


def music_inference_model(LSTM_cell, densor, n_values=78, n_a=64, Ty=100):
    """
    做一个用来预测音乐的模型
    :param LSTM_cell: 训练过的LSTM单元
    :param densor: 训练过的densor层
    :param n_values: 唯一编码数量
    :param n_a: LSTM隐藏层单元数量
    :param Ty: 生成的时间步数量
    :return:inference_model:用来预测音乐的模型
    """

    # 模型输入维度
    x0 = Input(shape=(1, n_values))

    # 定义a0,c0, 初始化隐藏态
    a0 = Input(shape=(n_a,), name="a0")
    c0 = Input(shape=(n_a,), name="c0")
    a = a0
    c = c0
    x = x0

    # 创建空列表存储输出
    outputs = []

    # 生成所有时间步的输出
    for t in range(Ty):
        # 单步传播
        a, _, c = LSTM_cell(x, initial_state=[a, c])
        #从LSTM的激活值中获取输出并保存
        out = densor(a)
        outputs.append(out)

        #选择最有可能的值并转化为独热编码并更新到x，x将传递给下一个时间步
        x = Lambda(one_hot)(out)

    inference_model = Model(inputs=[x0,a0,c0],outputs=outputs)
    return inference_model

def predict_and_sample(inference_model,x_initializer=x_initializer,a_initializer = a_initializer,c_initializer = c_initializer):
    """
    生成预测的独热编码序列
    :param inference_model: 用来走一个时间步的预测的模型
    :param x_initializer: 初始化的独热编码
    :param a_initializer: 初始化隐藏层输出
    :param c_initializer: 初始化记忆单元值
    :return:
    results:独热编码(Ty,78)
    indices:索引值矩阵(Ty,1)
    """
    #获取Y序列
    pred = inference_model.predict([x_initializer,a_initializer,c_initializer])

    #取预测的最大概率的项建立索引
    indices = np.argmax(pred,axis=-1)

    #将索引转换为独热编码
    results = to_categorical(indices,num_classes=78)

    return results, indices

