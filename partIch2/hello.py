import numpy as np
import matplotlib.pyplot as plt
from partIch2.lr_utils import load_dataset
from partIch2 import lr_functions as lf

# %%导入数据
# 将数据集导入
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

# 数据集里是个啥
print("训练集的数量：" + str(train_set_y.shape))
print("测试集的数量：" + str(test_set_y.shape))
print("训练图片数量/分辨率宽/分辨率高/通道数量" + str(train_set_x_orig.shape))
print("训练集的标签" + str(train_set_y.shape))
print("0号图片的第0个像素的RGB值" + str(train_set_x_orig[0, 0, 0, :]))

# 将数据集中的每张图片化为一维列向量
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
print("转换后训练集的维度：" + str(train_set_x_flatten.shape))
print("转换后测试集的维度：" + str(test_set_x_flatten.shape))

# 将数据集中的每张图片的每个像素转化为不会使sigmoid函数溢出的值(0~256会导致sigmoid函数溢出)
train_set_x = train_set_x_flatten / 255
test_set_x = test_set_x_flatten / 255

# %%构建不同学习率神经网络并测试
# learning_rates = [0.01, 0.001, 0.0001]
# modules = {}
# for i in learning_rates:
#     print("Learning rate is: " + str(i))
#     modules[str(i)] = lf.module(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=2000, learning_rate = i)
#     plt.plot(np.squeeze(modules[str(i)]["costs"]), label = str(modules[str(i)]["learning_rate"]))
# plt.ylabel('cost')
# plt.xlabel('iterations (per hundreds')
# plt.legend()
# plt.show()

d = lf.module(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=2000, learning_rate=0.005)

# %%成本函数图
costs = np.squeeze(d["costs"])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()
