from keras.models import load_model, Model
from keras.layers import Dense, Activation, Dropout, Input, LSTM, Reshape, Lambda, RepeatVector
from keras.initializers import glorot_uniform
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras import backend as K
import IPython
import sys
from music21 import *
from partVch1_3.grammar import *
from partVch1_3.qa import *
from partVch1_3.preprocess import *
from partVch1_3.music_utils import *
from partVch1_3.data_utils import *
from partVch1_3.network import *
import time
import numpy as np

# 创建模型
model = djmodel(Tx=30, n_a=64, n_values=78)
# 设置优化器
opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.01)  # decay控制学习率衰减
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

start_time = time.clock()
model.fit([X, a0, c0], list(Y), epochs=100)
end_time = time.clock()
minium = end_time - start_time
print("执行了：" + str(int(minium / 60)) + "分" + str(int(minium % 60)) + "秒")

# 获取预测模型并初始化参数
inference_model = music_inference_model(LSTM_cell, densor, n_values=78, n_a=64, Ty=50)
x_initializer = np.zeros((1, 1, 78))
a_initializer = np.zeros((1, n_a))
c_initializer = np.zeros((1, n_a))

# 进行预测
results, indices = predict_and_sample(inference_model, x_initializer, a_initializer, c_initializer)

# 记录音乐
out_stream = generate_music(inference_model)
