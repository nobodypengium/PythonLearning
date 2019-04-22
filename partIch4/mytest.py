import numpy as np
import h5py
import matplotlib.pyplot as plt
import partIch4.testCases
from partIch4.dnn_utils import *
from partIch4.lr_utils import *
from partIch4.network import *
import pickle

f = open("parameters.pkl", 'rb')
parameters = pickle.load(f)
image = Image.open("cat.jpg").convert("RGB").resize((64,64))
image_arr = np.array(image) / 255
image_arr = image_arr.reshape(-1, 1)
AL,cache = L_model_forward(image_arr, parameters)

prob = np.squeeze(AL)

if prob >= 0.5:
    image_predict = 1
    print("这是猫图"+str(prob))
else:
   image_predict = 0
   print("这并不是猫图"+str(prob))