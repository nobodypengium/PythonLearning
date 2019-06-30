from keras.models import Sequential #该模块允许多个网络层线性堆叠，也允许在已有的网络上加一些层
from keras.layers import Conv2D,ZeroPadding2D,Activation,Input,concatenate #concatenate函数用来从axis维度开始连接两个矩阵
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate #连接层
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras import backend as K

from IPython.display import SVG #画图
from keras.utils.vis_utils import model_to_dot #画图
from keras.utils import plot_model #画图

K.set_image_data_format('channels_first')

import time #时间操作转换
import cv2 #图像处理
import os
import numpy as np
import sys
from numpy import genfromtxt
import pandas as pd #数据结构 数据处理
import tensorflow as tf
import partIVch4_1.fr_utils as fr_utils #导入数据定义filter大小
from partIVch4_1.inception_blocks_v2 import * #定义网络模块

np.set_printoptions(threshold=sys.maxsize) #输出数组的时候完全输出，不需要省略号将中间数据省略


#%% 获取欲训练模型
FRModel = faceRecoModel(input_shape=(3,96,96))
print("参数数量 = " + str(FRModel.count_params()))
