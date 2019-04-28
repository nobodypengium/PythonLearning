import numpy as np
import h5py
import tensorflow as tf
from tensorflow.python.framework import ops
import partIIch3.tf_utils as tf_utils
import time
from partIIch3.network import *
import matplotlib.pyplot as plt


X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = tf_utils.load_dataset()