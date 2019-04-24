import partIIch1.chk_utils as chk_utils
import partIIch1.network as network

# %%测试一维线性的梯度
x, theta = 2, 4
difference = chk_utils.gradient_check(x, theta)

# %%测试维度检查
layers_dims = [4, 5, 3, 1]
parameters = network.initialize_parameters_he(layers_dims)
grads = chk_utils.backward_propagation()
# 没找到那么大的四维数据集
