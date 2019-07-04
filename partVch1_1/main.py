import numpy as np
import random
import time
from partVch1_1 import cllm_utils
from partVch1_1.network import *

#读入数据集
data = open("dinos.txt","r").read()
data = data.lower()
chars = list(set(data))

#创建字典
char_to_ix = {ch:i for i, ch in enumerate(sorted(chars))}
ix_to_char = {i:ch for i,ch in enumerate(sorted(chars))}

# 获取大小信息
data_size, vocab_size = len(data), len(chars)

#训练并获取生成的恐龙名字
start_time = time.clock()
parameters = model(data,ix_to_char,char_to_ix,num_iterations=3500)
end_time = time.clock()
minimum = end_time - start_time
print("执行了：" + str(int(minimum/60)) + "分" + str(int(minimum%60)) + "秒")
