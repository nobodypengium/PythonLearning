import numpy as np
import matplotlib as plt
import scipy.io
import math
import sklearn
import sklearn.datasets
from partIIch2.network import *

# %%测试用普通batch梯度下降更新参数
parameters, grads, learning_rate = testCase.update_parameters_with_gd_test_case()
parameters = update_parameters_with_gd(parameters, grads, learning_rate)

# %%测试分割mini_batch
X_assess, Y_assess, mini_batch_size = testCase.random_mini_batches_test_case()
mini_batch = random_mini_batches(X_assess,Y_assess,mini_batch_size)

# %%测试初始化速度
parameters = testCase.initialize_velocity_test_case()
v = initialize_velocity(parameters)

# %%测试动量梯度下降法更新参数
parameters,grads,v = testCase.update_parameters_with_momentum_test_case()
parameters,v = update_parameters_with_momentum(parameters,grads,v,beta=0.9,learning_rate=0.01)

# %%测试adam的初始化
parameters = testCase.initialize_adam_test_case()
V,S = initialize_adam(parameters)

# %%测试adam更新参数
parameters,grads,V,S = testCase.update_parameters_with_adam_test_case()
parameters,V,S = update_parameters_with_adam(parameters,grads,V,S,t=2)
