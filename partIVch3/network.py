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

from partIVch3.yad2k.models.keras_yolo import yolo_head, yolo_boxes_to_corners, preprocess_true_boxes, yolo_loss, \
    yolo_body

import partIVch3.yolo_utils as yolo_utils


def yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold=0.6):
    """
    过滤对象和分类的置信度，还有个作用是减少数据量，没有锚框的格子直接给扔了
    :param box_confidence: 所有锚框的p_c
    :param boxes: 所有锚框的位置
    :param box_class_probs: 所有锚框所有对象的检测概率
    :param threshold: 预测概率，高于它的预测保留
    :return:
    返回的时候我们忽略数组中(19,19,5)的部分，那一部分编程掩码了，我们并不关心锚框出现在哪个格子中，我们关心的是这个锚框应该画在什么位置，上面标上置信度
    """

    # 锚框得分
    box_scores = box_confidence * box_class_probs  # (19,19,5,80)

    # 最大值锚框的索引及其对应分数，以下两个变量为(19,19,5)
    box_classes = K.argmax(box_scores, axis=-1)  # 应该是一个格子只能找到一个锚框
    box_class_scores = K.max(box_scores, axis=-1)  # axis=-1指倒数第一个维度，以此类推，这里面每个数都是某个格子中最有可能出现的class的概率，还有这锚框里框了个啥

    # 创建掩码(19,19,5)
    filtering_mask = (box_class_scores >= threshold)

    # 使用掩码，只保留为True的锚框 [0,1,2,3] mask:[t,f,f,f] -> [0]，该步结束后，那些置信率太低的锚框被扔掉了
    scores = tf.boolean_mask(box_class_scores, filtering_mask)
    boxes = tf.boolean_mask(boxes, filtering_mask)
    classes = tf.boolean_mask(box_classes, filtering_mask)

    return scores, boxes, classes


def iou(box1, box2):
    """

    :param box1: 第一个锚框(x1,y1,x2,y2)(左上角坐标，右下角坐标）
    :param box2: 同上
    :return:
    """

    # 获得相交矩形的四个点
    xi1 = np.maximum(box1[0], box2[0])
    yi1 = np.maximum(box1[1], box2[1])
    xi2 = np.minimum(box1[2], box2[2])
    yi2 = np.minimum(box1[3], box2[3])
    inter_area = (xi2 - xi1) * (yi2 - yi1)

    # 计算并集
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    # IOU
    iou = inter_area / union_area

    return iou


def yolo_non_max_suppression(scores, boxes, classes, max_boxes=10, iou_threshold=0.5):
    """
    最大值抑制
    :param scores: 所有得分高于阈值的锚框的分数(?,)
    :param boxes: 这些锚框的位置(?,4)
    :param classes: 这些锚框的分类
    :param max_boxes: 预测锚框数量的最大值
    :param iou_threshold: IOU的阈值，高于这个阈值的认为是重叠的应当扔掉
    :return:
    """
    max_boxes_tensor = K.variable(max_boxes, dtype="int32")  # 在tf里处理数据，变量类型
    K.get_session().run(tf.variables_initializer([max_boxes_tensor]))

    # 使用非最大值抑制获取保留锚框的索引，这个里面就实现了循环直到所有边框都被遍历那个过程了
    nms_indices = tf.image.non_max_suppression(boxes, scores, max_boxes, iou_threshold)

    # 根据刚才获取的索引，从输入中把需要的锚框给拿出来
    scores = K.gather(scores, nms_indices)
    boxes = K.gather(boxes, nms_indices)
    classes = K.gather(classes, nms_indices)

    return scores, boxes, classes


def yolo_eval(yolo_outputs, image_shape=(720., 1280.), max_boxes=10, score_threshold=0.6, iou_threshold=0.5):
    """
    输入一张图，过CNN，输出了YOLO编码(yolo_outputs)，将这编码转换为非最大值抑制下的预测框，包括分数、位置，类
    :param yolo_outputs:一张图片通过CNN得到的输出，包括box_confidence(19,19,5,1),box_xy(19,19,5,2),box_wh(19,19,5,80
    )
    :param image_shape:输入图像的维度(2,)
    :param max_boxes:一张图片最多输出多少个锚框
    :param score_threshold:置信度阈值
    :param iou_threshold:交并比阈值
    :return:
    scores:每个预测的可能性
    boxes:每个锚框的位置
    classes:每个锚框框出的东西是个啥
    """
    # 拆解YOLO编码
    box_confidence, box_xy, box_wh, box_class_probs = yolo_outputs

    # 中心点+宽高表示的编码转化为左上右下顶点表示的编码
    boxes = yolo_boxes_to_corners(box_xy, box_wh)

    # 置信度分值过滤
    scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs, score_threshold)

    # 缩放锚框，以适应输入的720p图像
    boxes = yolo_utils.scale_boxes(boxes, image_shape)

    # 非最大值抑制
    scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes, max_boxes, iou_threshold)

    return scores, boxes, classes


