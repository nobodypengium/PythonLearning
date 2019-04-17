import numpy as np
import h5py
import matplotlib.pyplot as plt
import partIch4.testCases
from partIch4.dnn_utils import *
import partIch4.lr_utils
from PIL import Image


# %% 初始化两层网络
def initialize_parameters(n_x, n_h, n_y):
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    parameters = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2
    }

    return parameters


# %% 对于深度的神经网络的节点初始化
def initialize_parameters_deep(layers_dims):  # layers_dims秩为1数组，layers_dims[l]包含第l层节点的数量
    np.random.seed(3)
    parameters = {}
    L = len(layers_dims)

    for l in range(1, L):  # 假设L=4，这个就是1，2，3
        parameters["W" + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) / np.sqrt(
            layers_dims[l - 1])  # 用除以平方根代替*0.01防止梯度消失或梯度爆炸
        parameters["b" + str(l)] = np.zeros((layers_dims[l], 1))  # zeros写法的诡异错误，要套两层括号

    return parameters



# %% 前向传播
# ·线性部分：针对每一层计算Z
# ·激活部分：针对每一层，对Z应用激活函数
# ·组合成针对多层神经网络的函数

# 线性激活激活部分（一个节点的前向传播）
def linear_activation_forward(A_prev, W, b, activation):
    Z = np.dot(W, A_prev) + b  # E:setting an array element with a sequence.，返回值写错了对应错矩阵了

    if activation == "sigmoid":
        A, activation_cache = sigmoid(Z)
    elif activation == "relu":
        A, activation_cache = relu(Z)

    cache = ((A_prev, W, b), Z)
    return A, cache


# 多层的前向传播
def L_model_forward(X, parameters):
    caches = []  # 把前向传播中涉及到的每层的AWbZ存着，反向传播便于计算

    A = X  # 假设L=4，下面这个L就是3，下面for循环弄完了实际上还剩一个，这个L跟前面的L差1
    L = len(parameters) // 2  # "//"符号是整除，保留整数部分，每层有两个参数，取得层数

    for l in range(1, L):  # 循环左开右闭
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters["W" + str(l)], parameters["b" + str(l)], "relu")
        caches.append(cache)

    AL, cache = linear_activation_forward(A, parameters["W" + str(L)], parameters["b" + str(L)], "sigmoid")  # 把差的那个L给补上
    caches.append(cache)

    return AL, caches


# %%计算成本函数

def compute_cost(AL, Y):
    m = Y.shape[1]
    cost = -np.sum(np.multiply(np.log(AL), Y) + np.multiply(np.log(1 - AL), 1 - Y)) / m
    cost = np.squeeze(cost)  # 这里要返回一个数不是矩阵

    return cost


# %%反向传播
# ·激活部分，得到dZ
# ·线性部分，根据已有求导公式，计算dW,db,dA_prev参数
# ·组合成针对多层神经网络的函数

# 线性部分，已知dZ，计算每一个节点内部的动作
def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = np.dot(dZ, A_prev.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db


# 线性激活部分
def linear_activation_backward(dA, cache, activation="relu"):
    linear_cache, activation_cache = cache

    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)  # 这个cache里就一个Z
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db


# 整合以应对深层网络
def L_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches)

    dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    current_cache = caches[L - 1]  # 秩为1的数组从0数
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache,
                                                                                                 "sigmoid")

    for l in reversed(range(L - 1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 2)], current_cache,
                                                                    "relu")  # l+2因为数组从0数和下一个
        grads["dA" + str(l + 1)] = dA_prev_temp  # 向左
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads


# %% 更新参数
def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2

    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]

    return parameters


# %%整合神经网络
# 两层神经网络
def two_layer_model(X, Y, layers_dims, learning_rate=0.075, num_iterations=3000, print_cost=False, isPlot=True):
    np.random.seed(1)
    grads = {}
    costs = []
    (n_x, n_h, n_y) = layers_dims

    parameters = initialize_parameters(n_x, n_h, n_y)

    W1, b1, W2, b2 = parameters["W1"], parameters["b1"], parameters["W2"], parameters["b2"]


    for i in range(num_iterations):
        AL,caches = L_model_forward(X, parameters)  #前向传播
        cost = compute_cost(AL, Y)  #计算成本
        grads = L_model_backward(AL,Y,caches) #后向传播
        parameters = update_parameters(parameters,grads,learning_rate) #更新参数
        if i % 100 == 0:
            costs.append(cost)
            if print_cost:
                print("第{0:d}次的迭代成本值为{1:g}".format(i,cost))

    if isPlot:
        plt.plot(np.squeeze(costs))
        plt.ylabel("cost")
        plt.xlabel("iterations (per hundred")
        plt.title("Learning rage = " + str(learning_rate))
        plt.show()

    return parameters

#深层神经网络
def L_layer_model(X,Y,layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost = False, isPlot=True):
    np.random.seed(1)
    costs = []

    parameters = initialize_parameters_deep(layers_dims) #初始化变量

    for i in range(num_iterations):
        AL, caches = L_model_forward(X,parameters) #前向传播
        cost = compute_cost(AL,Y) #计算参数
        grads = L_model_backward(AL,Y,caches) #计算梯度
        parameters = update_parameters(parameters,grads,learning_rate) #新的参数
        if i % 100 == 0:
            costs.append(cost)
            if(print_cost):
                print("第{0:d}次迭代的成本函数是：{1:g}".format(i, cost))

    if isPlot:
        plt.plot(np.squeeze(costs))
        plt.ylabel("cost")
        plt.xlabel("iterations (per hundred)")
        plt.title("Learning rate = " + str(learning_rate))

    return parameters

# %% 预测
def predict(X,Y,parameters):
    m = X.shape[1]
    n = len(parameters) // 2
    Y_prediction= np.zeros((1,m))

    AL, cache = L_model_forward(X,parameters)

    for i in range (0,AL.shape[1]):
        if AL[0,i] > 0.5:
            Y_prediction[0,i] = 1

    print("准确度为：" + str((1 - np.sum(np.abs(Y - AL))/m)*100) + "%")

    return Y_prediction

# %%分析错误标记
def print_mislabeled_images(classes,X,Y,P):
    A = P + Y
    mislabeled_indices = np.asarray(np.where(A == 1)) #第index个图片是错误标记的
    plt.rcParams['figure.figsize'] = (40.0, 40.0)

    num_images = len(mislabeled_indices[0])
    for i in range(num_images):
        index = mislabeled_indices[1][i]

        plt.subplot(4,num_images/2,i + 1) #一共几行，一共几列，第几个图
        plt.imshow(X[:,index].reshape(64,64,3), interpolation='nearest')#显示某一张图片
        plt.axis('off')
        plt.title("Prediction: \n" + classes[int(P[0,index])].decode("utf-8") + "\nClass: \n" + classes[int(Y[0,index])].decode("utf-8"))

# %%判断某个自己的图片是不是猫

def predict_my_image(path,parameters):
    image = Image.open(path).convert("RGB").resize((64,64))
    image_arr = np.array(image)
    image_arr = image_arr.reshape(-1,1)
    AL,cache = L_model_forward(image_arr, parameters)

    prob = np.squeeze(AL)

    if prob >= 0.5:
        image_predict = 1
        print("这是猫图"+str(prob))
    else:
        image_predict = 0
        print("这并不是猫图"+str(prob))

    return image_predict