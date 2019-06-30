import time
start_time = time.clock()
from keras import backend as K
K.set_image_data_format('channels_first')
from partIVch4_1.inception_blocks_v2 import *
from partIVch4_1.network import *

#导入网络
FRModel = faceRecoModel(input_shape=(3,96,96))
#编译网络
FRModel.compile(optimizer='adam',loss=triplet_loss,metrics=['accuracy'])
#加载权值
fr_utils.load_weights_from_FaceNet(FRModel)
#结束时间
end_time = time.clock()
#计算时差
minium = end_time - start_time

print("执行了: " + str(int(minium/60))+"分"+str(int(minium%60))+"秒")