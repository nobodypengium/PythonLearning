from keras.callbacks import LambdaCallback
from keras.models import Model,load_model,Sequential
from keras.layers import Dense,Activation,Dropout,Input,Masking
from keras.layers import LSTM
from keras.utils.data_utils import get_file
from keras.preprocessing.sequence import pad_sequences
from partVch1_2.shakespeare_utils import * #在导入这个包的时候直接运行导入模型的语句
import sys
import io

#绘制图像
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model

# print_callback = LambdaCallback(on_epoch_end=on_epoch_end) # 调用之前训练好的模型
# model.fit(x,y,batch_size=128,epochs=1,callbacks=[print_callback]) # 再跟着之前训练的模型训练一代

# generate_output()

plot_model(model,to_file='shakespeare.svg')
SVG(model_to_dot(model).create(prog='dot',format='svg'))

