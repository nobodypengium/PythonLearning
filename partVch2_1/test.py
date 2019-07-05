import numpy as np
from partVch2_1 import emo_utils
import emoji
import matplotlib.pyplot as plt
from partVch2_1.network import *

word_to_index, index_to_word, word_to_vec_map = emo_utils.read_glove_vecs('data/glove.6B.50d.txt')

#%% 测试emoji输出
# X_train, Y_train = emo_utils.read_csv('data/train_emoji.csv')
# X_test, Y_test = emo_utils.read_csv('data/test.csv')
#
# maxLen = len(max(X_train,key=len).split())#最长的单词的长度
# index = 3
# print(X_train[index], emo_utils.label_to_emoji(Y_train[index]))

avg = sentence_to_avg("Morrocan couscous is my favorite dish", word_to_vec_map)
print("avg = ", avg)