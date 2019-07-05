import numpy as np
from partVch2_1 import emo_utils
import emoji
import matplotlib.pyplot as plt
from partVch2_1.network import *

word_to_index, index_to_word, word_to_vec_map = emo_utils.read_glove_vecs('data/glove.6B.50d.txt')
X_train, Y_train = emo_utils.read_csv('data/train_emoji.csv')
X_test, Y_test = emo_utils.read_csv('data/test.csv')
pred, W, b = model(X_train, Y_train, word_to_vec_map)
