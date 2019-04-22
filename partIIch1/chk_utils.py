import numpy as np
import partIIch1.gc_utils as gc_utils


# %% 一维线性测试
## 通过前向传播，进行双边检测
## 求导数
## 求得误差
def forward_propagation(x, theta):
    J = np.dot(theta, x)
    return J


def backward_propagation(x, theta):
    dtheta = x
    return dtheta


def gradient_check(x, theta, epsilon=1e-7):
    thetaplus = theta + epsilon
    thetaminus = theta - epsilon
    J_plus = forward_propagation(x, theta + epsilon)  # dy/dx 中 x的地位等同于这里的theta，theta是自变量，J是因变量，x是系数
    J_minus = forward_propagation(x, theta - epsilon)
    gradapprox = (J_plus - J_minus) / (2 * epsilon)

    grad = backward_propagation(x, theta)

    # 求“距离”
    numerator = np.linalg.norm(grad - gradapprox)  # 求范数，分子
    denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)  # 分母
    difference = numerator / denominator

    if difference < 1e-7:
        print("梯度正常")
    else:
        print("梯度超出阈值")

    return difference


# %%高维测试，在测试的时候不要开dropout和L2
def forward_propagation_n(X, Y, parameters):
    m = X.shape[1]
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]

    Z1 = np.dot(W1, X) + b1
    A1 = gc_utils.relu(Z1)

    Z2 = np.dot(W2, A1) + b2
    A2 = gc_utils.relu(Z2)

    Z3 = np.dot(W3, A2) + b3
    A3 = gc_utils.sigmoid(Z3)

    # 计算成本
    logprob = np.multiply(-np.log(A3), Y) + np.multiply(-np.log(1 - A3), 1 - Y)
    cost = (1 / m) * np.sum(logprob)

    cache = (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3)

    return cost, cache


def backward_propagation(X, Y, cache):
    m = X.shape[1]
    (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache

    # 反向传播
    dZ3 = A3 - Y
    dW3 = (1 / m) * np.dot(dZ3, A2.T)
    db3 = (1/m)*np.sum(dZ3,axis=1,keepdims=True)

    dA2 = np.dot(W3.T, dZ3)
    dZ2 = np.multiply(dA2,np.int64(A2>0))
    dW2 = (1/m)*np.dot(dZ2,A1.T)
    db2 = (1/m)*np.sum(dZ2,axis=1,keepdims=True)

    dA1 = np.dot(W2.T,dZ2)
    dZ1 = np.multiply(dA1,np.int64(A1>0))
    dW1 = (1/m)*np.dot(dZ1,A1.T)
    db1 = (1/m)*np.sum(dZ1,axis=1,keepdims=True)

    grads = {"dZ3":dZ3,"dW3":dW3,"db3":db3
            "dA2":dA2,"dZ2":dZ2,"dW2":dW2,"db2":db2
            "dA1":dA1,"dZ1":dZ1,"dW1":dW1,"db1":db1
    }

    return grads

#参数看起来比一维的多了因为有些是用于前向传播和反向传播的
def gradient_check_n(parameters,X,Y,epsilon=1e-7):