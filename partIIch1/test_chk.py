from partIIch1.chk_utils import *

# %%测试一维线性的梯度
x, theta = 2, 4
difference = gradient_check(x, theta)
