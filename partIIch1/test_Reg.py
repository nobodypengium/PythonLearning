import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import pylab
import partIIch1.init_utils as init_util
import partIIch1.reg_utils as reg_util
import partIIch1.gc_utils as gc_util
from partIIch1.network import *

train_X,train_Y,test_X,test_Y = reg_utils.load_2D_dataset(is_plot=False)

# %%进行预测
costs_without_reg, parameters_without_reg = model_reg(train_X,train_Y,print_cost=False)
costs_with_L2_reg, parameters_with_L2_reg = model_reg(train_X,train_Y,lambd=0.7,print_cost=False)
costs_with_dropout_reg, parameters_with_dropout_reg = model_reg(train_X, train_Y, keep_prob=0.86, learning_rate=0.3,print_cost=False)

# %%输出正则化/不使用正则化的准确率
print("不使用正则化：训练集：")
prediction_train_with_dropout_reg_train = reg_utils.predict(train_X,train_Y,parameters_without_reg)
print("不使用正则化：测试集：")
prediction_train_with_dropout_reg_test = reg_utils.predict(test_X,test_Y,parameters_without_reg)
print("使用L2正则化：训练集：")
prediction_train_with_dropout_reg_train = reg_utils.predict(train_X,train_Y,parameters_with_L2_reg)
print("使用L2正则化：测试集：")
prediction_train_with_dropout_reg_test = reg_utils.predict(test_X,test_Y,parameters_with_L2_reg)
print("随机删除节点：训练集：")
prediction_train_with_dropout_reg_train = reg_utils.predict(train_X,train_Y,parameters_with_dropout_reg)
print("随机删除节点：测试集：")
prediction_train_with_dropout_reg_test = reg_utils.predict(test_X,test_Y,parameters_with_dropout_reg)

# %%成本曲线对比
fig = pylab.figure(1)
fig.tight_layout(h_pad=3.0)
plt.subplot(2, 2, 1)
plt.plot(costs_without_reg, label='no')
plt.plot(costs_with_L2_reg, label='L2')
plt.plot(costs_with_dropout_reg, label='dropout')
plt.legend()
plt.ylabel("cost")
plt.xlabel("iterations (per thousand)")
plt.title("costs by different regulation method")

# %%不同正则化的预测值
# 没有正则化
plt.subplot(2, 2, 2)
reg_utils.load_2D_dataset(is_plot=True)
axes = plt.gca()
axes.set_xlim([-0.75, 0.40])
axes.set_ylim([-0.75, 0.65])
reg_utils.plot_decision_boundary(lambda x: reg_utils.predict_dec(parameters_without_reg,x.T),train_X,train_Y)
plt.title("no regulation")

# L2正则化
plt.subplot(2, 2, 3)
reg_utils.load_2D_dataset(is_plot=True)
axes = plt.gca()
axes.set_xlim([-0.75, 0.40])
axes.set_ylim([-0.75, 0.65])
reg_utils.plot_decision_boundary(lambda x:reg_utils.predict_dec(parameters_with_L2_reg,x.T),train_X, train_Y)
plt.title("L2 regulation")

# dropout 正则化
plt.subplot(2, 2, 4)
reg_utils.load_2D_dataset(is_plot=True)
axes = plt.gca()
axes.set_xlim([-0.75, 0.40])
axes.set_ylim([-0.75, 0.65])
reg_utils.plot_decision_boundary(lambda x:reg_utils.predict_dec(parameters_with_dropout_reg,x.T),train_X, train_Y)
plt.title("dropout regulation")