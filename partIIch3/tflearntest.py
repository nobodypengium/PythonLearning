import numpy as np
import h5py
import matplotlib as plt
import tensorflow as tf
from tensorflow.python.framework import ops
import partIIch3.tf_utils as tf_utils
import time
import os
from partIIch3.tflearn import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 忽略由于CPU支持AVX2引起的警告

# %%线性函数
# print("result = " + str(linear_function()))

# %%sigmoid函数
result, loss = sigmoid(0)
print("sigmoid(0) = " + str(result) + "loss = " + str(loss))

# %%独热编码
labels = np.array([[1, 2, 3, 0, 2, 1]])
one_hot = one_hot_matrix(labels,4)
print(str(one_hot))

# %%全1编码
print("ones = " + str(ones((3,3))))
