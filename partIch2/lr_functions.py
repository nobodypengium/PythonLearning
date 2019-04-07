import numpy as np


# sigmoid激活函数
def sigmoid(z):
    s = (1 / (1 + np.exp(-z)))
    return s


def cost_func(A, Y):
    m = Y.shape[1]
    cost = -(1 / m) * np.sum(Y * np.log(A) + ((1 - Y) * np.log(1 - A)))
    return cost


# 初始化系数
def initialize_with_zero(dim):
    w = np.zeros((dim, 1))
    b = 0
    return w, b


def propagate(w, b, X, Y):
    m = X.shape[1]  # 样本数量

    # 正向传播
    A = sigmoid(np.dot(w.T, X) + b)
    cost = cost_func(A, Y)
    cost = np.squeeze(cost)

    # 反向传播
    dz = A - Y
    dw = (1 / m) * np.dot(X, dz.T)
    db = (1 / m) * np.sum(dz)

    # 确保计算的数据维度正确
    assert (dw.shape == w.shape)
    assert (db.dtype == float)
    assert (cost.shape == ())

    # 创建字典，保存梯度
    grads = {
        "dw": dw,
        "db": db
    }
    return grads, cost


def optimize(w, b, X, Y, num_iterations, learning_rate):
    # 优化期间每次迭代的成本函数值
    costs = []

    for i in range(num_iterations):

        grads, cost = propagate(w, b, X, Y)
        dw = grads["dw"]
        db = grads["db"]

        w = w - learning_rate * dw
        b = b - learning_rate * db

        if i % 100 == 0:
            costs.append(cost)

    params = {
        "w": w,
        "b": b
    }

    grads = {
        "dw": dw,
        "db": db
    }

    return (params, grads, costs)


def predict(w, b, X):
    # 初始化测试集个数m，存储预测结果变量Y_prediction，再确保一下w是需要的大小
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(-1, 1)

    # 算激活函数，也就是结果的概率，比0.5大的就是1，小的就是0，二分分类
    A = sigmoid(np.dot(w.T, X) + b)
    for i in range(A.shape[1]):
        if A[0, i] > 0.5:
            Y_prediction[0, i] = 1

    # 保证Y_prediction正确格式
    assert (Y_prediction.shape == (1, m))

    return Y_prediction


def module(X_train, Y_train, X_test, Y_test, num_iterations, learning_rate):
    # 初始化wb为0
    w, b = initialize_with_zero(X_train.shape[0])
    # 梯度下降
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate)

    w, b = parameters["w"], parameters["b"]

    Y_prediction_train = predict(w, b, X_train)
    Y_prediction_test = predict(w, b, X_test)

    print("训练集准确性：", str(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100), "%")
    print("测试集准确性：", str(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100), "%")

    data = {
        "costs": costs,
        "Y_prediction_test": Y_prediction_test,
        "Y_prediction_train": Y_prediction_train,
        "w": w,
        "b": b,
        "learning_rate": learning_rate,
        "num_iterations": num_iterations
    }

    return data
