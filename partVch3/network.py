from keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply
from keras.layers import RepeatVector, Dense, Activation, Lambda
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import load_model, Model
import keras.backend as K
import numpy as np

from faker import Faker
import random
from tqdm import tqdm
from babel.dates import format_date
from partVch3.nmt_utils import *
import matplotlib.pyplot as plt

# 读入数据集
m = 10000
Tx = 30
Ty = 10
dataset, human_vocab, machine_vocab, inv_machine_vocab = load_dataset(m)
X, Y, Xoh, Yoh = preprocess_data(dataset, human_vocab, machine_vocab, Tx, Ty)
# 将共享层定义为全局变量以免参数更新
repeater = RepeatVector(Tx)
concatenator = Concatenate(axis=-1)
densor1 = Dense(10, activation="tanh")
densor2 = Dense(1, activation="relu")
activator = Activation(softmax, name="attention_weights")
dotor = Dot(axes=1)
# post-attention的共享层
n_a = 32
n_s = 64
post_activation_LSTM_cell = LSTM(n_s, return_state=True)
output_layer = Dense(len(machine_vocab), activation=softmax)


def one_step_attention(a, s_prev):
    """
    针对post-attention LSTM的某一层计算注意力权重并由此生成上下文向量
    :param a: 来自pre-attention 的 Bi-LSTM的隐藏层输出，(m,Tx,2*n_a)
    :param s_prev:前一个post-attention的隐藏层输出
    :return:上下文向量context<t>
    """
    s_prev = repeater(s_prev)
    concat = concatenator([a, s_prev])
    e = densor1(concat)
    energies = densor2(e)
    alphas = activator(energies)
    context = dotor([alphas, a])
    return context


def model(Tx, Ty, n_a, n_s, human_vocab_size, machine_vocab_size):
    """
    输入人类的时间表示输出机器可以理解的时间表示，包含一个单向LSTM和一个双向LSTM
    :param Tx:
    :param Ty:
    :param n_a:
    :param n_s:
    :param human_vocab_size:
    :param machine_vocab_size:
    :return:  model:keras model
    """

    X = Input(shape=(Tx, human_vocab_size))
    s0 = Input(shape=(n_s,), name='s0')
    c0 = Input(shape=(n_s,), name='c0')
    # 初始化
    s = s0
    c = c0
    outputs = []
    # 创建一个Bi-LSTM
    a = Bidirectional(LSTM(n_a, return_sequences=True), name='bidirectional_1')(X)

    # 迭代Ty，每次需要用到Bi-LSTM中的值
    for t in range(Ty):
        # 计算context作为s<t>的输入
        context = one_step_attention(a, s)
        # 把context丢入post-attention的LSTM模型中
        s, _, c = post_activation_LSTM_cell(context, initial_state=[s, c])
        # Dense -> output
        out = output_layer(s)
        outputs.append(out)

    # 创造并返回模型
    model = Model(inputs=[X, s0, c0], outputs=outputs)
    return model
