import numpy as np
import matplotlib.pyplot as plt
from partIch3.testCases import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from partIch3.planar_utils import *

np.random.seed(1)
X, Y = load_planar_dataset()