from partIVch3.network import *

# %%测试过滤置信率低的锚框
# with tf.Session() as test_a:
#     box_confidence = tf.random_normal([19,19,5,1],mean=1,stddev=4,seed=1)#用于从服从指定正态分布的数值中取出指定个数的值。
#     boxes = tf.random_normal([19,19,5,4],mean=1,stddev=4,seed=1)
#     box_class_probs = tf.random_normal([19,19,5,80],mean=1,stddev=4,seed=1)
#     scores,boxes,classes=yolo_filter_boxes(box_confidence,boxes,box_class_probs,threshold=0.5)
#
#     print("scores[2] = " + str(scores[2].eval()))
#     print("boxes[2] = " + str(boxes[2].eval()))
#     print("classes[2] = " + str(classes[2].eval()))
#     print("scores.shape = " + str(scores.shape))
#     print("boxes.shape = " + str(boxes.shape))
#     print("classes.shape = " + str(classes.shape))#输出带?指这个维度不确定，需要由上下文确定
#
#     test_a.close()

# %%测试计算IOU
# box1 = (2, 1, 4, 3)
# box2 = (1, 2, 3, 4)
#
# print("IOU = " + str(iou(box1, box2)))

# %%测试非最大值抑制
# with tf.Session() as test_b:
#     scores = tf.random_normal([54,],mean=1,stddev=4,seed=1)
#     boxes = tf.random_normal([54,4],mean=1,stddev=4,seed=1)
#     classes = tf.random_normal([54,],mean=1,stddev=4,seed=1)
#     scores, boxes, classes = yolo_non_max_suppression(scores,boxes,classes)
#
#     print("scores[2] = " + str(scores[2].eval()))
#     print("boxes[2] = " + str(boxes[2].eval()))
#     print("classes[2] = " + str(classes[2].eval()))
#     print("score.shapes = " + str(scores.shape))
#     print("boxes.shapes = " + str(boxes.shape))
#     print("classes.shape = " + str(classes.shape))
#
#     test_b.close()

# %%测试从一个图片通过CNN的输出中提取出锚框
with tf.Session() as test_c:
    yolo_outputs = (tf.random_normal([19, 19, 5, 1], mean=1, stddev=4, seed=1),
                    tf.random_normal([19, 19, 5, 2], mean=1, stddev=4, seed=1),
                    tf.random_normal([19, 19, 5, 2], mean=1, stddev=4, seed=1),
                    tf.random_normal([19, 19, 5, 80], mean=1, stddev=4, seed=1))
    scores, boxes, classes = yolo_eval(yolo_outputs)

    print("scores[2] = " + str(scores[2].eval()))
    print("boxes[2] = " + str(boxes[2].eval()))
    print("classes[2] = " + str(classes[2].eval()))
    print("scores.shape = " + str(scores.eval().shape))
    print("boxes.shape = " + str(boxes.eval().shape))
    print("classes.shape = " + str(classes.eval().shape))

    test_c.close()
