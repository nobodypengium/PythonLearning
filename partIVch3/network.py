import argparse  # 提供方法来接收命令行传参、分割、给变量加tag、help等，测试如下，pycharm中用run/debug configurations设置传参
import os  # 文件 目录 路径
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import scipy.io  # 读取数据
import scipy.misc  # 读取图片
import numpy as np
import pandas as pd  # 数据结构 数据处理
import PIL  # 图像处理 展示 变换 分割 阈值等等
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Lambda, Conv2D
from keras.models import load_model, Model

from partIVch3.yad2k.models.keras_yolo import yolo_head, yolo_boxes_to_corners, preprocess_true_boxes, yolo_loss, yolo_body

import partIVch3.yolo_utils as yolo_utils

def yolo_filter_boxes(box_confidence,boxes,box_class_probs,threshold=0.6):
    """
    过滤对象和分类的置信度
    :param box_confidence: 所有锚框的p_c
    :param boxes: 所有锚框的位置
    :param box_class_probs: 所有锚框所有对象的检测概率
    :param threshold: 预测概率，高于它的预测保留
    :return:
    """

    #锚框得分
    box_scores = box_confidence * box_class_probs #(19,19,5,80)

    #最大值锚框的索引及其对应分数
    box_classes = K.argmax(box_scores,axis=-1) #应该是一个格子只能找到一个锚框
    box_class_scores = K.max(box_scores,axis=-1) #axis=-1指倒数第一个维度，以此类推