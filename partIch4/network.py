import numpy as np
import h5py
import matplotlib.pyplot as plt
import partIch4.testCases
from partIch4.dnn_utils import *
import partIch4.lr_utils

# %% 初始化两层网络
def initialize_parameters(n_x, n_h, n_y):
    W1 = np.random.randn(n_h,n_x)*0.01
    b1 = np.zeros((n_h,1))
    W2 = np.random.randn(n_y,n_h)*0.01
    b2 = np.zeros((n_y,1))

    parameters = {
        "W1":W1,
        "b1":b1,
        "W2":W2,
        "b2":b2
    }

    return parameters

