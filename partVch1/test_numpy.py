from partVch1.network_numpy import *

# %% 测试一个RNN模块的前向传播
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

# %% 测试RNN网络的前向传播
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

# 测试LSTM模块
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

# %% 测试LSTM网络
# np.random.seed(1)
# x = np.random.randn(3,10,7)
# a0 = np.random.randn(5,10)
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
#
# a,y,c,caches = lstm_forward(x,a0,parameters)

# %%测试单个时间步的反向传播
# np.random.seed(1)
# xt = np.random.randn(3, 10)
# a_prev = np.random.randn(5, 10)
# Waa = np.random.randn(5, 5)
# Wax = np.random.randn(5, 3)
# Wya = np.random.randn(2, 5)
# ba = np.random.randn(5, 1)
# by = np.random.randn(2, 1)
# parameters = {
#     "Waa": Waa,
#     "Wax": Wax,
#     "Wya": Wya,
#     "ba": ba,
#     "by": by
# }
#
# # 前向传播
# a_next, yt_pred, cache = rnn_cell_forward(xt, a_prev, parameters)
# # 反向传播
# da_next = np.random.randn(5, 10)
# gradients = rnn_cell_backward(da_next, cache)

#%% 对整个网络进行测试
# np.random.seed(1)
# x = np.random.randn(3,10,4)
# a0 = np.random.randn(5,10)
# Wax = np.random.randn(5,3)
# Waa = np.random.randn(5,5)
# Wya = np.random.randn(2,5)
# ba = np.random.randn(5,1)
# by = np.random.randn(2,1)
# parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "ba": ba, "by": by}
# a, y, caches = rnn_forward(x, a0, parameters)
# da = np.random.randn(5, 10, 4)
# gradients = rnn_backward(da, caches)
#
# print("gradients[\"dx\"][1][2] =", gradients["dx"][1][2])
# print("gradients[\"dx\"].shape =", gradients["dx"].shape)
# print("gradients[\"da0\"][2][3] =", gradients["da0"][2][3])
# print("gradients[\"da0\"].shape =", gradients["da0"].shape)
# print("gradients[\"dWax\"][3][1] =", gradients["dWax"][3][1])
# print("gradients[\"dWax\"].shape =", gradients["dWax"].shape)
# print("gradients[\"dWaa\"][1][2] =", gradients["dWaa"][1][2])
# print("gradients[\"dWaa\"].shape =", gradients["dWaa"].shape)
# print("gradients[\"dba\"][4] =", gradients["dba"][4])
# print("gradients[\"dba\"].shape =", gradients["dba"].shape)

#%% 测试单个时间步的LSTM
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
#
# a_next, c_next, yt, cache = lstm_cell_forward(xt, a_prev, c_prev, parameters)
#
# da_next = np.random.randn(5,10)
# dc_next = np.random.randn(5,10)
# gradients = lstm_cell_backward(da_next, dc_next, cache)
# print("gradients[\"dxt\"][1][2] =", gradients["dxt"][1][2])
# print("gradients[\"dxt\"].shape =", gradients["dxt"].shape)
# print("gradients[\"da_prev\"][2][3] =", gradients["da_prev"][2][3])
# print("gradients[\"da_prev\"].shape =", gradients["da_prev"].shape)
# print("gradients[\"dc_prev\"][2][3] =", gradients["dc_prev"][2][3])
# print("gradients[\"dc_prev\"].shape =", gradients["dc_prev"].shape)
# print("gradients[\"dWf\"][3][1] =", gradients["dWf"][3][1])
# print("gradients[\"dWf\"].shape =", gradients["dWf"].shape)
# print("gradients[\"dWi\"][1][2] =", gradients["dWi"][1][2])
# print("gradients[\"dWi\"].shape =", gradients["dWi"].shape)
# print("gradients[\"dWc\"][3][1] =", gradients["dWc"][3][1])
# print("gradients[\"dWc\"].shape =", gradients["dWc"].shape)
# print("gradients[\"dWo\"][1][2] =", gradients["dWo"][1][2])
# print("gradients[\"dWo\"].shape =", gradients["dWo"].shape)
# print("gradients[\"dbf\"][4] =", gradients["dbf"][4])
# print("gradients[\"dbf\"].shape =", gradients["dbf"].shape)
# print("gradients[\"dbi\"][4] =", gradients["dbi"][4])
# print("gradients[\"dbi\"].shape =", gradients["dbi"].shape)
# print("gradients[\"dbc\"][4] =", gradients["dbc"][4])
# print("gradients[\"dbc\"].shape =", gradients["dbc"].shape)
# print("gradients[\"dbo\"][4] =", gradients["dbo"][4])
# print("gradients[\"dbo\"].shape =", gradients["dbo"].shape)

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

a, y, c, caches = lstm_forward(x, a0, parameters)

da = np.random.randn(5, 10, 4)
gradients = lstm_backward(da, caches)

print("gradients[\"dx\"][1][2] =", gradients["dx"][1][2])
print("gradients[\"dx\"].shape =", gradients["dx"].shape)
print("gradients[\"da0\"][2][3] =", gradients["da0"][2][3])
print("gradients[\"da0\"].shape =", gradients["da0"].shape)
print("gradients[\"dWf\"][3][1] =", gradients["dWf"][3][1])
print("gradients[\"dWf\"].shape =", gradients["dWf"].shape)
print("gradients[\"dWi\"][1][2] =", gradients["dWi"][1][2])
print("gradients[\"dWi\"].shape =", gradients["dWi"].shape)
print("gradients[\"dWc\"][3][1] =", gradients["dWc"][3][1])
print("gradients[\"dWc\"].shape =", gradients["dWc"].shape)
print("gradients[\"dWo\"][1][2] =", gradients["dWo"][1][2])
print("gradients[\"dWo\"].shape =", gradients["dWo"].shape)
print("gradients[\"dbf\"][4] =", gradients["dbf"][4])
print("gradients[\"dbf\"].shape =", gradients["dbf"].shape)
print("gradients[\"dbi\"][4] =", gradients["dbi"][4])
print("gradients[\"dbi\"].shape =", gradients["dbi"].shape)
print("gradients[\"dbc\"][4] =", gradients["dbc"][4])
print("gradients[\"dbc\"].shape =", gradients["dbc"].shape)
print("gradients[\"dbo\"][4] =", gradients["dbo"][4])
print("gradients[\"dbo\"].shape =", gradients["dbo"].shape)