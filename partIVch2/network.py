import numpy as np
from keras import layers #kera网络层的相关函数，设置与获取权重，获取输入输出尺寸与数据
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, BatchNormalization, Flatten, Conv2D
from keras.models import Model #用于对模型中输入输出张量的信息进行操作
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot #绘图用
from IPython.display import SVG #交互式计算系统
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
import partIVch2.kt_utils

import keras.backend as K #后端 处理数据转换
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
