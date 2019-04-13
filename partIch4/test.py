import numpy as np
import h5py
import matplotlib.pyplot as plt
import partIch4.testCases
from partIch4.dnn_utils import *
from partIch4.lr_utils import *
from partIch4.network import *

np.random.seed(1)
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
train_set_x = train_set_x_flatten / 255
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
test_set_x = test_set_x_flatten / 255

n_x = train_set_x[0]
n_h = 4
n_y = train_set_y[0]

parameters = initialize_parameters(n_x, n_h, n_y)

print("W1: " + str(parameters["W1"].shape))
print("b1: " + str(parameters["b1"].shape))
print("W2: " + str(parameters["W2"].shape))
print("b2: " + str(parameters["b2"].shape))
