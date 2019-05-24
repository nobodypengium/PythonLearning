import numpy as np
from keras import layers #kera网络层的相关函数，设置与获取权重，获取输入输出尺寸与数据
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, BatchNormalization, Flatten, Conv2D
from keras.models import Model #Model类可以给定输入输出以创建模型
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

def model(input_shape):
    #定义TF中的placeholder
    X_input = Input(input_shape)

    #0填充
    X = ZeroPadding2D((3,3))(X_input)

    #Kera定义某项操作，与被操作的数，放在两个括号中
    #Conv -> 归一化 ->RELU，覆盖处理，不用在每一层创建新变量，使用X覆盖所有值
    X = Conv2D(32,(7,7),strides=(1,1),name='conv0')
    X = BatchNormalization(axis=3, name="bn0") #使得每个通道均值为0 标准差为1
    X = Activation('relu')(X)

    #最大值池化
    X = MaxPooling2D((2,2),name="max_pool")(X) #stride默认None为与pool_size相同

    #降维，以便全连接层输入，全连接层输出一维向量
    X = Flatten()(X)
    X = Dense(1,activation='sigmoid',name='fc')(X) #1是指输出维度

    #创建模型(一个模型类)
    model = Model(inputs = X_input,outputs=X,name='HappyModel')

    return model

def HappyModel(input_shape):
    #定义一个placeholder
    X_input = Input(input_shape)

    #使用0填充
    X = ZeroPadding2D((2,2))(X_input)#上下填充2，左右填充2

    #CONV->BN->RELU块
    X = Conv2D(32,(7,7),strides=(1,1),name='conv0')(X)
    X = BatchNormalization(axis=3,name='bn0')(X)
    X = Activation('relu')(X)

    #最大值池化
    X = MaxPooling2D((2,2),name='max_pool')(X)

    #降维以便输入+FC
    X = Flatten()(X)
    X = Dense(1,activation='sigmoid',name='fc')(X)

    #创建模型
    model = Model(inputs=X_input,outputs=X,name='HappyModel')

    return model
