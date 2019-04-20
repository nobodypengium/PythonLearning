import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import partIIch1.init_utils as init_util
import partIIch1.reg_utils as reg_util
import partIIch1.gc_utils as gc_util

#设置默认值，后面有些函数会覆盖
plt.rcParams['figure.figsize'] = (7.0,4.0)      #设置图片大小
plt.rcParams['image.interpolation'] = 'nearest' #设置插值方式
plt.rcParams['image.cmap'] = 'gray_r'           #设置图像外观

train_X, train_Y, test_X, test_Y = init_util.load_dataset(is_plot=True) #用sklern里面的函数随机生成带噪音的圆环

