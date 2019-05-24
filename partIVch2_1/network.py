import numpy as np
import tensorflow as tf
from keras import layers #layers里有对层的操作，比如建立一个卷积层，建立一个归一化层等
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D,BatchNormalization,Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalAveragePooling2D
from keras.models import Model, load_model #对模型整体进行操作，比如编译、预测、评估、存取等
from keras.preprocessing import image #读取图像
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras_applications.imagenet_utils import preprocess_input
from keras.utils.vis_utils import model_to_dot #画图用
from keras.utils import plot_model #还是画图用
from keras.initializers import glorot_uniform #初始化用参数从[-limit, limit]的均匀分布产生，其中limit为sqrt(6 / (fan_in + fan_out))

#以下全都是画图用的
import pydot
from IPython.display import SVG
import scipy.misc
from matplotlib.pyplot import imshow


import keras.backend as K #基本的类型转换啊之类的底层（相对于层）的操作
K.set_image_data_format('channels_last')
K.set_learning_phase(1) #测试模式

def identity_block(X,f,filters,stage,block):
    """
    一个跳过2个隐藏层的恒等快（输入输出维度相同）
    :param X: 输入的张量
    :param f: 中间那个fxf filter的维度
    :param filters: 过滤器数量
    :param stage: 命名
    :param block: 命名
    :return: X: 输出的张量
    """

    #定义命名规则
    conv_name_base = "res" + str(stage) + block + "_branch"
    bn_name_base = "bn" + str(stage) + block + "_branch"

    #获取每层的过滤器数量
    F1, F2, F3 = filters

    #捷径要传送的数据
    X_shortcut = X

    # X -> CONV -> BN -> ReLU -> CONV -> BN ->ReLu -> CONV -> BN -> + ->ReLu
    # | -----------------------------------------------------------↑

    #CONV
    X = Conv2D(filters = F1, kernel_size=(1,1), strides=(1,1), padding="valid",name = bn_name_base + "2a", kernel_initializer=glorot_uniform(0))(X) # glorot_uniform(0)Xavier初始化指定种子为0

    #BN
    X = BatchNormalization(axis=3,name=bn_name_base + "2b")(X)

    #Activation
    X = Activation("relu")(X)

    #CONV
    X = Conv2D(filters=F2,kernel_size=(f,f),strides=(1,1),padding="same",name=conv_name_base+"2b",kernel_initializer=glorot_uniform(seed=0))(X)

    #BN
    X = BatchNormalization(axis=3,name=bn_name_base + '2b')(X)

    #Activation
    X = Activation("relu")(X)

    #CONV
    X = Conv2D(filters=F3,kernel_size=(1,1),padding="valid",name=conv_name_base+"2c",kernel_initializer=glorot_uniform(seed=0))
    