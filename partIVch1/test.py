import numpy as np
import h5py
import matplotlib.pyplot as plt
from partIVch1.network_numpy import *

plt.rcParams['figure.figsize'] = (5.0, 4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(1)

# %%测试两层通道的填充
# x = np.random.randn(4,3,3,2)
# x_paded = zero_pad(x,2)
# print("x.shape = ",x.shape)
# print("x_paded.shape = ",x_paded.shape)
# print("x[1,1] = ",x[1,1])
# print("x_paded[1,1] = ",x_paded[1,1])
#
# fig , axarr = plt.subplots(1,2)
# axarr[0].set_title('x')
# axarr[0].imshow(x[0,:,:,0])
# axarr[1].set_title('x_paded')
# axarr[1].imshow(x_paded[0,:,:,0])

#%% 测试卷积
# a_slice_prev = np.random.randn(4,4,3)
# W = np.random.randn(4,4,3)
# b = np.random.randn(1,1,1)
#
# Z = conv_single_step(a_slice_prev,W,b)
#
# print("Z = "+str(Z))


#%%测试前向传播
# A_prev = np.random.randn(10,4,4,3)
# W = np.random.randn(2,2,3,8)
# b = np.random.randn(1,1,1,8)
##f存在W里
# hparameters={
#     "pad":2,
#     "stride":1
# }
#
# Z,cache_conv = conv_forward(A_prev,W,b,hparameters)
#
# print("np.mean(Z) = " + str(np.mean(Z)))
# print("cache_conv[0][1][2][3] = " + str(cache_conv[0][1][2][3]))

#%% 测试池化

A_prev = np.random.randn(2,4,4,3)

#p=0
hparameters = {
    "f":4,
    "stride":1
}

A,cache = pool_forward(A_prev,hparameters,mode="max")
print("mode max A = " + str(A))
A,cache = pool_forward(A_prev,hparameters,mode="average")
print("mode average A = "+str(A))


