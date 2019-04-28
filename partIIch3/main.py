import numpy as np
import h5py
import tensorflow as tf
from tensorflow.python.framework import ops
import partIIch3.tf_utils as tf_utils
import time
from partIIch3.network import *
import matplotlib.pyplot as plt


X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = tf_utils.load_dataset()

#处理导入数据
X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0],-1).T
X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0],-1).T

#归一化
X_train = X_train_flatten/255
X_test = X_test_flatten/255

#转化为独热矩阵
Y_train = tf_utils.convert_to_one_hot(Y_train_orig,6)
Y_test = tf_utils.convert_to_one_hot(Y_test_orig,6)

start_time = time.clock()
parameters = model(X_train,Y_train,X_test,Y_test)
end_time = time.clock()
print("CPU的执行时间 = " + str(end_time - start_time) + "秒")


