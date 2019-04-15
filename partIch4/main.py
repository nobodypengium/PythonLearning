import numpy as np
import h5py
import matplotlib.pyplot as plt
import partIch4.testCases
from partIch4.dnn_utils import *
from partIch4.lr_utils import *
from partIch4.network import *

# %% 输入数据
np.random.seed(1)
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
train_set_x = train_set_x_flatten / 255
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
test_set_x = test_set_x_flatten / 255

n_x = train_set_x.shape[0]
n_y = train_set_y.shape[0]

np.random.seed(1)
layer_dims = [n_x, 20, 7, 5, 1]

# %%两层网络
# parameters = two_layer_model(train_set_x,train_set_y,layer_dims,learning_rate=0.0075,num_iterations=2500,print_cost=True,isPlot=True)

# %%多层网络
parameters = L_layer_model(train_set_x, train_set_y, layer_dims, learning_rate=0.0075, num_iterations=2500, print_cost=True, isPlot=True)

# %%预测
print("=====训练集=====")
train_set_y_predict = predict(train_set_x,train_set_y,parameters)
print("=====测试集=====")
test_set_y_predict = predict(test_set_x,test_set_y,parameters)

# %% 错误标记的图片
#print_mislabeled_images(classes,test_set_x,test_set_y,test_set_y_predict)

# %%自己的图片
