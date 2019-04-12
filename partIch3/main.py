import numpy as np
import matplotlib.pyplot as plt
from partIch3.testCases import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from partIch3.planar_utils import *
from partIch3.network import *

np.random.seed(1)
X, Y = load_planar_dataset()
# 为了统一测试不同数据集，把这个不一样的数据集进行维度转换
X = X.T
Y = np.squeeze(Y)
flower = (X, Y)
noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure = load_extra_datasets()
datasets = {
    "flower": flower,
    "noisy_circles": noisy_circles,
    "noisy_moons": noisy_moons,
    "blobs": blobs,
    "gussian_quantiles": gaussian_quantiles
}

dataset = "gussian_quantiles"
X, Y = datasets[dataset]
X = X.T
Y = Y.reshape(1, Y.shape[0])

if dataset == "blobs":
    Y = Y % 2 #blobs这个有四种颜色，Y有四个值

plt.scatter(X[0, :], X[1, :], c=np.squeeze(Y), s=40, cmap=plt.cm.Spectral)

parameters = nn_model(X, Y, 50, 5000, True)
# lambda返回一个函数 输入x，返回一个预测函数，这么写使得parameters在这里就可以被定义，在planar_util里面就直接输个x进去就行了
plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)

predictions = predict(parameters, X)
print("成本函数: %d" % float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100) + "%")
plt.show()
