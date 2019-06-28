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
    X = Conv2D(filters = F1, kernel_size=(1,1), strides=(1,1), padding="valid",name = conv_name_base + "2a", kernel_initializer=glorot_uniform(0))(X) # glorot_uniform(0)Xavier初始化指定种子为0

    #BN
    X = BatchNormalization(axis=3,name=bn_name_base + "2a")(X)

    #Activation
    X = Activation("relu")(X)

    #CONV
    X = Conv2D(filters=F2,kernel_size=(f,f),strides=(1,1),padding="same",name=conv_name_base+"2b",kernel_initializer=glorot_uniform(seed=0))(X)

    #BN
    X = BatchNormalization(axis=3,name=bn_name_base + '2b')(X)

    #Activation
    X = Activation("relu")(X)

    #CONV
    X = Conv2D(filters=F3,kernel_size=(1,1),padding="valid",name=conv_name_base+"2c",kernel_initializer=glorot_uniform(seed=0))(X)

    #BN
    X = BatchNormalization(axis=3,name=bn_name_base+"2c")(X)

    #捷径
    X = Add()([X,X_shortcut])

    #Activation
    X = Activation("relu")(X)

    return X

def convolutional_block(X,f,filters,stage,block,s = 2):
    """
    实现卷积块，应对维度发生变化的层
    :param X:输入的tensor变量
    :param f:非1x1卷积层的filter的大小
    :param filters:三层每层有几个filter
    :param stage:命名用
    :param block:命名用
    :param s:第三层（最后一层）使用的步幅
    :return:卷积块的输出
    """

    #命名规则
    conv_name_base = "res" + str(stage) + block + "_branch"
    bn_name_base = "bn" + str(stage) + block + "_branch"

    #每层过滤器水量
    F1,F2,F3 = filters

    #保存输入数据用于捷径
    X_shortcut = X

    # X -> CONV -> BN -> ReLU -> CONV -> BN ->ReLu -> CONV -> BN -> + ->ReLu
    # | ---------------------- CONV -> BN --------------------------↑

    #CONV->BN->ReLU
    X = Conv2D(filters = F1, kernel_size=(1,1), strides=(s,s),padding="valid",name=conv_name_base + "2a", kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + "2a")(X)
    X = Activation("relu")(X)

    #CONV->BN->ReLU
    X = Conv2D(filters = F2, kernel_size=(f,f),strides=(1,1),padding="same",name=conv_name_base + "2b", kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3,name=bn_name_base + "2b")(X)
    X = Activation("relu")(X)

    #CONV->BN->ReLU
    X = Conv2D(filters = F3, kernel_size=(1,1),strides=(1,1),padding="valid",name=conv_name_base + "2c", kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + "2c")(X)

    #捷径 A->CONV->BN
    X_shortcut = Conv2D(filters = F3, kernel_size=(1,1), strides=(s,s), padding="valid", name=conv_name_base+"1", kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3,name=bn_name_base+"1")(X_shortcut)

    # +
    X = Add()([X,X_shortcut])
    X = Activation("relu")(X)

    return X

def ResNet50(input_shape=(64,64,3),classes=6):
    """
    CONV2D -> BN -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> FC -> Softmax
    :param input_shape: 输入数据集的维度
    :param classes: 分类数
    :return:
    """

    #创建占位符
    X_input = Input(input_shape)

    #0填充
    X = ZeroPadding2D((3,3))(X_input) #将输入上下左右都添加3排0

    #第一部分:CONV2D -> BN -> RELU -> MAXPOOL
    X = Conv2D(filters=64, kernel_size=(7,7),strides=(2,2),name="conv1",kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3,name="bn_conv1")(X)
    X = Activation("relu")(X)
    X = MaxPooling2D(pool_size=(3,3),strides=(2,2))(X)

    #第二部分:CONVBLOCK -> IDBLOCK*2
    X = convolutional_block(X,f=3,filters=[64,64,256],stage=2,block="a",s=1)
    X = identity_block(X, f=3, filters=[64,64,256], stage=2,block="b")
    X = identity_block(X, f=3, filters=[64,64,256], stage=2, block="c")

    #第三部分:CONVBLOCK -> IDBLOCK*3
    X = convolutional_block(X, f=3, filters=[128,128,512], stage=3,block="a",s=2)
    X = identity_block(X, f=3, filters=[128,128,512], stage=3, block="b")
    X = identity_block(X, f=3, filters=[128, 128, 512], stage=3, block="c")
    X = identity_block(X, f=3, filters=[128, 128, 512], stage=3, block="d")

    #第四部分:CONVBLOCK -> IDBLOCK * 5
    X = convolutional_block(X, f=3, filters = [256,256,1024], stage=4, block="a", s=2)
    X = identity_block(X, f=3, filters=[256,256,1024], stage=4, block="b")
    X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block="c")
    X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block="d")
    X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block="e")
    X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block="f")

    #第五部分:CONVBLOCK -> IDBLOCK*2
    X = convolutional_block(X, f=3, filters=[512,512,2048], stage=5, block="a", s=2)
    X = identity_block(X, f=3, filters=[512,512,2048],stage=5,block="b")
    X = identity_block(X, f=3, filters=[512, 512, 2048], stage=5, block="c")

    #第六部分:AVGPOOL -> FC -> Softmax
    X = AveragePooling2D(pool_size=(2,2),padding="same")(X)
    X = Flatten()(X)
    X = Dense(classes,activation="softmax",name="fc"+str(classes),kernel_initializer=glorot_uniform(seed=0))(X)#

    model = Model(inputs=X_input,outputs=X,name="ResNet50")

    return model