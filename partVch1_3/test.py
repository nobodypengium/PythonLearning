from keras.models import load_model,Model
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

X, Y, n_values, indices_values = load_music_utils()
print('shape of X:', X.shape) # (60, 30, 78) m = 60 个样本 Tx = 30 30个时间步 78 每个时间步有78个可能的输出
print('number of training examples:', X.shape[0])
print('Tx (length of sequence):', X.shape[1])
print('total # of unique values:', n_values)
print('Shape of Y:', Y.shape)