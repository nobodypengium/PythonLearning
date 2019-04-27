import numpy as np
import matplotlib as plt
import scipy.io
import math
import sklearn
import sklearn.datasets
from partIIch2.network import *
import pylab

# %%测试用普通batch梯度下降更新参数
parameters, grads, learning_rate = testCase.update_parameters_with_gd_test_case()
parameters = update_parameters_with_gd(parameters, grads, learning_rate)

# %%测试分割mini_batch
X_assess, Y_assess, mini_batch_size = testCase.random_mini_batches_test_case()
mini_batch = random_mini_batches(X_assess,Y_assess,mini_batch_size)

# %%测试初始化速度
parameters = testCase.initialize_velocity_test_case()
v = initialize_velocity(parameters)

# %%测试动量梯度下降法更新参数
parameters,grads,v = testCase.update_parameters_with_momentum_test_case()
parameters,v = update_parameters_with_momentum(parameters,grads,v,beta=0.9,learning_rate=0.01)

# %%测试adam的初始化
parameters = testCase.initialize_adam_test_case()
V,S = initialize_adam(parameters)

# %%测试adam更新参数
parameters,grads,V,S = testCase.update_parameters_with_adam_test_case()
parameters,V,S = update_parameters_with_adam(parameters,grads,V,S,t=2)

#%% 加载数据集
train_X, train_Y = opt_utils.load_dataset(is_plot=False)

#%% 梯度下降测试
layers_dims=[train_X.shape[0],5,2,1]
parameters_gd,costs_gd = model(train_X,train_Y,layers_dims,optimizer="gd",is_plot=False)
parameters_momentum,costs_momentum = model(train_X,train_Y,layers_dims,optimizer="momentum",is_plot=False)
parameters_adam,costs_adam = model(train_X,train_Y,layers_dims,optimizer="adam",is_plot=False)

# %%绘制分类图
plt.subplot(2,2,1)
plt.plot(costs_gd, label='gd')
plt.plot(costs_momentum, label='momentum')
plt.plot(costs_adam, label='adam')
plt.legend()
plt.xlabel("epoch (per hundred)")
plt.ylabel("cost")
plt.title("cost by different optimiser")
plt.subplot(2,2,2)
predictions = opt_utils.predict(train_X,train_Y,parameters_gd)
plot_prediction("Model with GD optimization",train_X,train_Y,parameters_gd)
plt.subplot(2,2,3)
predictions = opt_utils.predict(train_X,train_Y,parameters_momentum)
plot_prediction("Model with momentum optimization",train_X,train_Y,parameters_momentum)
plt.subplot(2,2,4)
prediction = opt_utils.predict(train_X,train_Y,parameters_adam)
plot_prediction("Model with adam optimization",train_X,train_Y,parameters_adam)
