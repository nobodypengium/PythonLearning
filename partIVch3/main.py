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

from partIVch3.network import *

def predict(sess, image_file, is_show_info=True, is_plot=True):
    """
    运行sess中的计算图来预测image_file的边框，打印出预测的图与信息
    :param sess:会话，包含计算图
    :param image_file:图像名称
    :param is_show_info:
    :param is_plot:
    :return:
    """

    # 图像预处理
    image, image_data = yolo_utils.preprocess_image("images/" + image_file, model_image_size=(608, 608))

    # 运行会话
    out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes],
                                                  feed_dict={yolo_model.input: image_data, K.learning_phase(): 0})

    #预测信息
    if is_show_info:
        print("在" + str(image_file) + "中找到了" + str(len(out_boxes)) + "个锚框")

    #边框颜色
    colors = yolo_utils.generate_colors(class_names)

    #绘制边框
    yolo_utils.draw_boxes(image,out_scores,out_boxes,out_classes,class_names,colors)

    #保存绘制了边界的图
    image.save(os.path.join("out",image_file),quality=100)

    #打印绘制了边界的图
    if is_plot:
        output_image = scipy.misc.imread(os.path.join("out",image_file))
        plt.imshow(output_image)

    return out_scores, out_boxes, out_classes


with tf.Session() as sess:
    class_names = yolo_utils.read_classes("model_data/coco_classes.txt")
    anchors = yolo_utils.read_anchors("model_data/yolo_anchors.txt")
    image_shape = (720., 1280.)

    yolo_model = load_model("model_data/yolov2.h5")
    # yolo_model.summary()

    yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
    scores, boxes, classes = yolo_eval(yolo_outputs, image_shape)
    # out_scores, out_boxes, out_classes = predict(sess, "test.jpg")

    for i in range(1,121):
        #获取文件名
        filename = str(i).zfill(4) + ".jpg"
        print("当前文件：" + str(filename))

        # 开始绘制，不打印信息，不绘制图
        out_scores, out_boxes, out_classes = predict(sess, filename, is_show_info=False, is_plot=False)

    print("绘制完成！")

    sess.close()
