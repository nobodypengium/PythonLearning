import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import pylab
import partIIch1.init_utils as init_util
import partIIch1.reg_utils as reg_util
import partIIch1.gc_utils as gc_util
from partIIch1.network import *

# 设置默认值，后面有些函数会覆盖
plt.rcParams['figure.figsize'] = (7.0, 4.0)  # 设置图片大小
plt.rcParams['image.interpolation'] = 'nearest'  # 设置插值方式
plt.rcParams['image.cmap'] = 'gray_r'  # 设置图像外观

# %%看看数据集
train_X, train_Y, test_X, test_Y = init_util.load_dataset(is_plot=False)  # 用sklern里面的函数随机生成带噪音的圆环

# %%测试初始化函数
layers_dims = np.squeeze([2, 4, 1])
parameters_zeros = initialize_parameters_zeros(layers_dims)
parameters_random = initialize_parameters_random(layers_dims)
parameters_he = initialize_parameters_he(layers_dims)

# %%用不同的初始化参数通过神经网络学习参数
parameters_zeros, costs_zeros = model(train_X, train_Y, initialization="zeros")
parameters_random, costs_random = model(train_X, train_Y, initialization="random")
parameters_he, costs_he = model(train_X, train_Y, initialization="he")
# %%用学习到的参数预测，输出准确率
print("zeros训练集：")
prediction_train_zeros = init_util.predict(train_X, train_Y, parameters_zeros)
print("zeros测试集：")
prediction_test_zeros = init_util.predict(test_X, test_Y, parameters_zeros)
print("random训练集：")
prediction_train_random = init_util.predict(train_X, train_Y, parameters_random)
print("random测试集：")
prediction_test_random = init_util.predict(test_X, test_Y, parameters_random)
print("he训练集：")
prediction_train_he = init_util.predict(train_X, train_Y, parameters_he)
print("he测试集：")
prediction_test_he = init_util.predict(test_X, test_Y, parameters_he)
# %%画成本曲线图
pylab.figure(1)
plt.plot(costs_zeros, label='zeros')
plt.plot(costs_random, label='random')
plt.plot(costs_he, label='he')
plt.legend()
plt.ylabel('cost')
plt.xlabel('iterations(per hundreds)')
plt.title("costs by different initialization")
# %%画分区图
# 用0初始化
pylab.figure(2)
# init_util.load_dataset(is_plot=True)
init_util.plot_decision_boundary(lambda x: init_utils.predict_dec(parameters_zeros, x.T), train_X, train_Y)
plt.title("zeros")
# 随机初始化
pylab.figure(3)
# init_util.load_dataset(is_plot=True)
init_util.plot_decision_boundary(lambda x: init_utils.predict_dec(parameters_random, x.T), train_X, train_Y)
plt.title("random")
# 抑制梯度异常初始化
pylab.figure(4)
# init_util.load_dataset(is_plot=True)
init_util.plot_decision_boundary(lambda x: init_utils.predict_dec(parameters_he, x.T), train_X, train_Y)
plt.title("he")