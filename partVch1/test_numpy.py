from partVch1.network_numpy import *

#%% 测试一个RNN模块的前向传播
# np.random.seed(1)
# xt = np.random.randn(3,10)
# a_prev = np.random.randn(5,10)
# Waa = np.random.randn(5,5)
# Wax = np.random.randn(5,3)
# Wya = np.random.randn(2,5)
# ba = np.random.randn(5,1)
# by = np.random.randn(2,1)
# parameters = {
#     "Waa":Waa,
#     "Wax":Wax,
#     "Wya":Wya,
#     "ba":ba,
#     "by":by
# }
#
# a_next, yt_pred, cache = rnn_cell_forward(xt,a_prev,parameters)
#
# print("a_next[4] = ", a_next[4])
# print("a_next.shape = ", a_next.shape)
# print("yt_pred[1] = ", yt_pred[1])
# print("yt_pred.shape = ", yt_pred.shape)

#%% 测试RNN网络的前向传播
# np.random.seed(1)
# x = np.random.randn(3,10,4)
# a0 = np.random.randn(5,10)
# Waa = np.random.randn(5,5)
# Wax = np.random.randn(5,3)
# Wya = np.random.randn(2,5)
# ba = np.random.randn(5,1)
# by = np.random.randn(2,1)
# parameters = {
#     "Waa":Waa,
#     "Wax":Wax,
#     "Wya":Wya,
#     "ba":ba,
#     "by":by
# }
#
# a, y_pred, caches = rnn_forward(x,a0,parameters)

#测试LSTM模块
# np.random.seed(1)
# xt = np.random.randn(3,10)
# a_prev = np.random.randn(5,10)
# c_prev = np.random.randn(5,10)
# Wf = np.random.randn(5, 5+3)
# bf = np.random.randn(5,1)
# Wi = np.random.randn(5, 5+3)
# bi = np.random.randn(5,1)
# Wo = np.random.randn(5, 5+3)
# bo = np.random.randn(5,1)
# Wc = np.random.randn(5, 5+3)
# bc = np.random.randn(5,1)
# Wy = np.random.randn(2,5)
# by = np.random.randn(2,1)
#
# parameters = {"Wf": Wf, "Wi": Wi, "Wo": Wo, "Wc": Wc, "Wy": Wy, "bf": bf, "bi": bi, "bo": bo, "bc": bc, "by": by}
# a_next, c_next, yt, cache = lstm_cell_forward(xt, a_prev, c_prev, parameters)

#%% 测试LSTM网络
np.random.seed(1)
x = np.random.randn(3,10,7)
a0 = np.random.randn(5,10)
Wf = np.random.randn(5, 5+3)
bf = np.random.randn(5,1)
Wi = np.random.randn(5, 5+3)
bi = np.random.randn(5,1)
Wo = np.random.randn(5, 5+3)
bo = np.random.randn(5,1)
Wc = np.random.randn(5, 5+3)
bc = np.random.randn(5,1)
Wy = np.random.randn(2,5)
by = np.random.randn(2,1)

parameters = {"Wf": Wf, "Wi": Wi, "Wo": Wo, "Wc": Wc, "Wy": Wy, "bf": bf, "bi": bi, "bo": bo, "bc": bc, "by": by}

a,y,c,caches = lstm_forward(x,a0,parameters)