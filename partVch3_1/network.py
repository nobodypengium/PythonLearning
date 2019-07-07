# 合成音频所需
import numpy as np
from pydub import AudioSegment  # 音频处理
import random
import sys
import io
import os
import glob  # 文件名匹配
import IPython
from partVch3_1.td_utils import *
# 搭建网络所需
from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Activation, Dropout, Input, Masking, TimeDistributed, LSTM, Conv1D #TimeDistributed将一个层应用于多个时间步，每层参数相同
from keras.layers import GRU, Bidirectional, BatchNormalization, Reshape
from keras.optimizers import Adam


# 读入生成的训练集
X = np.load("./XY_train/X.npy")
Y = np.load("./XY_train/Y.npy")
X_dev = np.load("./XY_dev/X_dev.npy")
Y_dev = np.load("./XY_dev/Y_dev.npy")


def model(input_shape):
    """
    创建一个keras模型，输入输入的形状，输出一串序列，其中1指示一个触发词的结束
    :param input_shape:
    :return:
    """

    X_input = Input(shape=input_shape)

    # CONV -> BN -> RELU -> Dropout
    X = Conv1D(196, 15, strides=4)(X_input)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Dropout(0.8)(X)

    # GRU->Dropout->BN
    X = GRU(128, return_sequences=True)(X)  # GRU (use 128 units and return the sequences)
    X = Dropout(0.8)(X)  # dropout (use 0.8)
    X = BatchNormalization()(X)  # Batch normalization

    # GRU->Dropout->BN->Dropout
    X = GRU(128, return_sequences=True)(X)  # GRU (use 128 units and return the sequences)
    X = Dropout(0.8)(X)  # dropout (use 0.8)
    X = BatchNormalization()(X)  # Batch normalization
    X = Dropout(0.8)(X)  # dropout (use 0.8)

    # Step 4: Dense -> sigmoid
    X = TimeDistributed(Dense(1, activation="sigmoid"))(X)  # time distributed  (sigmoid)

    model = Model(inputs=X_input, outputs=X)

    return model

Tx = 5511  # 频谱图模型输入时间步
n_freq = 101  # 频谱图模型频率数
model = load_model('./models/tr_model.h5')
opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.01)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])
model.fit(X, Y, batch_size = 5, epochs=1)


def detect_triggerword(filename):
    plt.subplot(2, 1, 1)

    x = graph_spectrogram(filename)
    x = x.swapaxes(0, 1)
    x = np.expand_dims(x, axis=0)
    predictions = model.predict(x)

    plt.subplot(2, 1, 2)
    plt.plot(predictions[0, :, 0])
    plt.ylabel('probability')
    plt.show()
    return predictions


chime_file = "audio_examples/chime.wav"


def chime_on_activate(filename, predictions, threshold):
    audio_clip = AudioSegment.from_wav(filename)
    chime = AudioSegment.from_wav(chime_file)
    Ty = predictions.shape[1]
    # 遍历所有输出
    consecutive_timesteps = 0
    for i in range(Ty):
        consecutive_timesteps += 1
        # 在输出为1的概率大于阈值的第一次添加响声
        if predictions[0, i, 0] > threshold and consecutive_timesteps > 75:
            audio_clip = audio_clip.overlay(chime, position=((i / Ty) * audio_clip.duration_seconds) * 1000)
            consecutive_timesteps = 0

    audio_clip.export("chime_output.wav", format='wav')