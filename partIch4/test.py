import numpy as np
import h5py
import matplotlib.pyplot as plt
import partIch4.testCases as testCases
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
n_h = 4
n_y = train_set_y.shape[0]

# %% 测试初始化两层神经网络
parameters = initialize_parameters(n_x, n_h, n_y)

print("W1: " + str(parameters["W1"].shape))
print("b1: " + str(parameters["b1"].shape))
print("W2: " + str(parameters["W2"].shape))
print("b2: " + str(parameters["b2"].shape))

# %% 测试初始化深层神经网络

layers_dims = [n_x, 4, 3, n_y]
parameters_deep = initialize_parameters_deep(layers_dims)
for l in range(1, layers_dims.__len__()):
    print("W" + str(l) + ": " + str(parameters_deep["W" + str(l)].shape))
    print("b" + str(l) + ": " + str(parameters_deep["b" + str(l)].shape))

 # %% 测试前向传播
AL, caches = L_model_forward(train_set_x, parameters_deep) #这里直接从documentation里读各个矩阵大小

 # %% 测试成本函数
cost = compute_cost(AL, train_set_y)
print("cost: {0:g}".format(cost))

 # %% 测试反向传播
grads = L_model_backward(AL,train_set_y,caches) #直接从documentation里看

 # %% 测试更新参数
parameters = update_parameters(parameters,grads,learning_rate=0.01)
