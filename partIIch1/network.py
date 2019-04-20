import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import partIIch1.init_utils as init_util
import partIIch1.reg_utils as reg_util
import partIIch1.gc_utils as gc_util

def initialize_parameters_zeros(layers_dims):
    parameters = {}

    L = len(layers_dims)
    for i in range(1,L):
        #TODO:

def initialize_parameters_random(layers_dims):

def initialize_parameters_he(layers_dims):

def model(X,Y,learning_rate=0.01,num_iterations=15000,print_cost=True,initialization = "he", isPlot = True):
    """
    先用源码提供的通用工具包试着写一下，没问题再换自己的
    :param X:
    :param Y:
    :param learning_rate:
    :param num_iterations:
    :param print_cost:
    :param initialization:
    :param isPlot:
    :return:
    """
    grads={}    #最后一次的梯度，只是记录，对于实际模型没有作用
    costs=[]    #成本函数，只是记录用于输出，对实际模型没有作用

    m = X.shape[1]
    layer_dims = [X.shape[0],10,5,1]

    #选择初始化参数的类型
    if initialization == "zeros":
        parameters =