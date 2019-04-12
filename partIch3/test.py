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

# %%测试逻辑回归准确性
# plt.scatter(X[0, :], X[1, :], c=np.squeeze(Y), s=40, cmap=plt.cm.Spectral)

# clf = sklearn.linear_model.LogisticRegressionCV()
# clf.fit(X.T, Y.T)

# plot_decision_boundary(lambda x: clf.predict(x), X, np.squeeze(Y))
# plt.title("Logistic Regression")
# LR_predictions = clf.predict(X.T)
# print(
#     "逻辑回归的准确性：%d" % float((np.dot(Y, LR_predictions) + np.dot(1 - Y, 1 - LR_predictions)) / float(Y.size) * 100) + "%")

# %%测试神经网络的搭建是否正确 layer_size(X,Y)
n_x, n_h, n_y = layer_size(X, Y)
print("输入层节点数量：n_x = %d" % n_x)
print("隐藏层节点数量：n_h = %d" % n_h)
print("输出层节点数量：n_y = %d" % n_y)

# %%测试初始化参量 initialize_parameters(n_x,n_h,n_y)
parameters = initialize_parameters(n_x, n_h, n_y)
print("W1 = ")
print(str(parameters["W1"]))
print("b1 = ")
print(str(parameters["b1"]))
print("W2 = ")
print(str(parameters["W2"]))
print("b2 = ")
print(str(parameters["b2"]))

# %%测试前向传播 forward_propagation
cache = forward_propagation(X, parameters)
print("Z1 = ")
print(str(cache["Z1"].shape))
print("A1 = ")
print(str(cache["A1"].shape))
print("Z2 = ")
print(str(cache["Z2"].shape))
print("A2 = ")
print(str(cache["A2"].shape))

# %%测试成本函数 compute_cost(
cost = compute_cost(cache["A2"], Y)
print("cost = %f" % cost)

# %%测试反向传播函数 backward_propagation
grads = backward_propagation(parameters, cache, X, Y)
print("dW1: " + str(grads["dW1"].shape))
print("db1: " + str(grads["db1"].shape))
print("dW2: " + str(grads["dW2"].shape))
print("db2: " + str(grads["db2"].shape))

# %%测试梯度下降 update_parameters
parameters = update_parameters(parameters,grads,1.2)
print("W1 = ")
print(str(parameters["W1"]))
print("b1 = ")
print(str(parameters["b1"]))
print("W2 = ")
print(str(parameters["W2"]))
print("b2 = ")
print(str(parameters["b2"]))

# %%测试神经网络 nn_model
parameters = nn_model(X,Y,4,100,False)
print("W1 = ")
print(str(parameters["W1"]))
print("b1 = ")
print(str(parameters["b1"]))
print("W2 = ")
print(str(parameters["W2"]))
print("b2 = ")
print(str(parameters["b2"]))
