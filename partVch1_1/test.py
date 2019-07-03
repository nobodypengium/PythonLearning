import numpy as np
import random
import time
from partVch1_1 import cllm_utils
from partVch1_1.network import *

#%% 测试抽样

# #读入数据集
# data = open("dinos.txt","r").read()
# data = data.lower()
# chars = list(set(data))
#
# #创建字典
# char_to_ix = {ch:i for i, ch in enumerate(sorted(chars))}
# ix_to_char = {i:ch for i,ch in enumerate(sorted(chars))}
#
# # 获取大小信息
# data_size, vocab_size = len(data), len(chars)
#
# np.random.seed(2)
# _, n_a = 20, 100
# Wax, Waa, Wya = np.random.randn(n_a, vocab_size), np.random.randn(n_a, n_a), np.random.randn(vocab_size, n_a)
# b, by = np.random.randn(n_a, 1), np.random.randn(vocab_size, 1)
# parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "b": b, "by": by}
#
#
# indices = sample(parameters, char_to_ix, 0)
# print("Sampling:")
# print("list of sampled indices:", indices)
# print("list of sampled characters:", [ix_to_char[i] for i in indices])

# %%测试优化
np.random.seed(1)
vocab_size, n_a = 27, 100
a_prev = np.random.randn(n_a, 1)
Wax, Waa, Wya = np.random.randn(n_a, vocab_size), np.random.randn(n_a, n_a), np.random.randn(vocab_size, n_a)
b, by = np.random.randn(n_a, 1), np.random.randn(vocab_size, 1)
parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "b": b, "by": by}
X = [12,3,5,11,22,3]
Y = [4,14,11,22,25, 26]

loss, gradients, a_last = optimize(X, Y, a_prev, parameters, learning_rate = 0.01)
print("Loss =", loss)
print("gradients[\"dWaa\"][1][2] =", gradients["dWaa"][1][2])
print("np.argmax(gradients[\"dWax\"]) =", np.argmax(gradients["dWax"]))
print("gradients[\"dWya\"][1][2] =", gradients["dWya"][1][2])
print("gradients[\"db\"][4] =", gradients["db"][4])
print("gradients[\"dby\"][1] =", gradients["dby"][1])
print("a_last[4] =", a_last[4])