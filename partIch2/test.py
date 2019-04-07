import numpy as np
import matplotlib.pyplot as plt
import h5py
from partIch2.lr_utils import load_dataset

# %% 导入数据集并预处理数据

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

train_set_x_orig_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_orig_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

train_set_x = train_set_x_orig_flatten / 255
test_set_x = test_set_x_orig_flatten / 255


# %%初始化w,b
def initialize_with_zeros(dim):
    w = np.zeros((dim,1))
    b = 0
    return w, b


# %% 激活函数
def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s


# %% 传播函数
def propagate(w, b, X, Y):
    m = X.shape[1]

    # 正向传播
    z = np.dot(w.T, X) + b
    A = sigmoid(z)
    cost = -(1 / m) * np.sum(Y * np.log(A) + ((1 - Y) * np.log(1 - A)))

    # 反向传播
    dz = A - Y
    dw = (1 / m) * np.dot(X, dz.T)
    db = (1 / m) * np.sum(dz)

    grads = {
        "dw": dw,
        "db": db,
    }
    return grads, cost


# %% 梯度下降函数
def optimize(w, b, X, Y, num_iterations, learning_rate):
    costs = []

    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)
        dw = grads["dw"]
        db = grads["db"]

        w = w - dw * learning_rate
        b = b - db * learning_rate

        if i % 100 == 0:
            costs.append(cost)

    params = {
        "w": w,
        "b": b,
    }

    return params, grads, costs


# %% 预测函数
def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(-1, 1)

    z = np.dot(w.T, X) + b
    A = sigmoid(z)
    for i in range(A.shape[1]):
        if A[0, i] > 0.5:
            Y_prediction[0, i] = 1

    assert (Y_prediction.shape == (1, m))

    return Y_prediction


# %% 测试预测准确度
def model(X_train, Y_train, X_test, Y_test, num_iteration, learning_rate):
    w, b = initialize_with_zeros(X_train.shape[0])
    params, grads, costs = optimize(w, b, X_train, Y_train, num_iteration, learning_rate)
    w, b = params["w"], params["b"]

    Y_train_prediction = predict(w, b, X_train)
    Y_test_prediction = predict(w, b, X_test)

    accuracy_train = 100 - (np.mean(np.abs(Y_train - Y_train_prediction)) * 100)
    accuracy_test = 100 - (np.mean(np.abs(Y_test - Y_test_prediction)) * 100)

    print("训练集准确率：" + str(accuracy_train) + "%")
    print("测试集准确率：" + str(accuracy_test) + "%")

    data = {
        "costs": costs,
        "learning_rate": learning_rate,
        "num_iteration": num_iteration,
        "w": w,
        "b": b
    }

    return data


# %%输出结果
data = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iteration=2000, learning_rate=0.005)
plt.plot(np.squeeze(data["costs"]))
plt.xlabel("num iteration")
plt.ylabel("cost")
plt.title("Learning rate =" + str(data["learning_rate"]))
plt.show()
