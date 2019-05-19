import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from tensorflow.python.framework import ops
import partIVch1.cnn_utils as cnn_utils
from partIVch1.network_tf import *

np.random.seed(1)

#读取并创建训练集和测试集
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = cnn_utils.load_dataset()
X_train = X_train_orig/255
X_test = X_test_orig/255
Y_train = cnn_utils.convert_to_one_hot(Y_train_orig,6).T #为什么还要转置？
Y_test = cnn_utils.convert_to_one_hot(Y_test_orig,6).T

_, _, parameters = model(X_train, Y_train, X_test, Y_test,num_epochs=150)