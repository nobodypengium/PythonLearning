import numpy as np
from pydub import AudioSegment  # 音频处理
import random
import sys
import io
import os
import glob  # 文件名匹配
import IPython
from partVch3_1.td_utils import *

# 以下键值都对应10s
_, data = wavfile.read("audio_examples/example_train.wav")  # (441000Hz,)
Tx = 5511  # 频谱图模型输入时间步
n_freq = 101  # 频谱图模型频率数
Ty = 1375  # 输出时间步数量
activates, negatives, backgrounds = load_raw_audio()


def get_random_time_segment(segment_ms):
    """
    随机选取一个时间段准备插入音频
    :param segment_ms:
    :return:
    """
    segment_start = np.random.randint(low=0, high=10000 - segment_ms)
    segment_end = segment_start + segment_ms - 1
    return (segment_start, segment_end)


def is_overlapping(segment_time, previous_segments):
    """
    遍历之前所有段，检查是否当前段与它们重叠
    :param segment_time:
    :param previous_segments:
    :return:
    """
    segment_start, segment_end = segment_time

    overlap = False
    # 遍历，若有重叠，置旗标为True
    for previous_start, previous_end in previous_segments:
        if segment_start >= previous_start and segment_start <= previous_end:
            overlap = True

    return overlap


def insert_audio_clip(background, audio_clip, previous_segments):
    """
    1.以ms为单位获取正确持续时间的随机时间段。
    2.确保时间段不与之前的任何时间段重叠。 如果它重叠，则返回步骤1并选择一个新的时间段。
    3.将新时间段添加到现有时间段列表中，以便跟踪您插入的所有时间段。
    4.叠加音频片段
    :param background:
    :param audio_clip:
    :param previous_segments:
    :return:
    """
    segment_ms = len(audio_clip)
    segment_time = get_random_time_segment(segment_ms)  # 1.以ms为单位获取正确持续时间的随机时间段。
    while is_overlapping(segment_time, previous_segments):  # 2.确保时间段不与之前的任何时间段重叠。 如果它重叠，则返回步骤1并选择一个新的时间段。
        segment_time = get_random_time_segment(segment_ms)
    previous_segments.append(segment_time)  # 3.将新时间段添加到现有时间段列表中，以便跟踪您插入的所有时间段。
    new_background = background.overlay(audio_clip, position=segment_time[0])  # 4.叠加音频片段
    return new_background, segment_time


def insert_ones(y, segment_end_ms):
    """
    更新正确输出y，在触发词结束后最多添加49个1
    :param y:
    :param segment_end_ms:
    :return:
    """
    segment_end_y = int(segment_end_ms * Ty / 10000.0)
    for i in range(segment_end_y + 1, segment_end_y + 51):
        if i < Ty:
            y[0, i] = 1
    return y


def create_training_example(background, activates, negatives):
    """
    1.将标签向量y初始化为numpy数组，shape （1，Ty）。
    2.将 existing segments的集合初始化为空列表。
    3.随机选择0到4个“activate”音频剪辑，然后将它们插入10秒剪辑。 还要在标签向量y中的正确位置插入标签。
    4.随机选择0到2个负音频剪辑，并将它们插入10秒剪辑中。
    :param background:
    :param activates:
    :param negatives:
    :return:
    """
    np.random.seed(18)
    background = background - 20  # 降低background的音频
    y = np.zeros((1, Ty))  # 1.将标签向量y初始化为numpy数组，shape （1，Ty）。
    previous_segments = []  # 2.将 existing segments的集合初始化为空列表。
    # 3.随机选择0到4个“activate”音频剪辑，然后将它们插入10秒剪辑。 还要在标签向量y中的正确位置插入标签。
    number_of_activates = np.random.randint(0, 5)
    random_indices = np.random.randint(len(activates), size=number_of_activates)
    random_activates = [activates[i] for i in random_indices]
    for random_activate in random_activates:
        background, segment_time = insert_audio_clip(background, random_activate, previous_segments)
        segment_start, segment_end = segment_time
        y = insert_ones(y, segment_end)
    # 4.随机选择0到2个负音频剪辑，并将它们插入10秒剪辑中。
    number_of_negatives = np.random.randint(0, 3)
    random_indices = np.random.randint(len(negatives), size=number_of_negatives)
    random_negatives = [negatives[i] for i in random_indices]
    for random_negative in random_negatives:
        background, _ = insert_audio_clip(background, random_negative, previous_segments)
    background = match_target_amplitude(background, -20.0)
    file_handle = background.export("train" + ".wav", format="wav")
    print("合成音频已输出")
    x = graph_spectrogram("train.wav")
    return x,y

# 读入生成的训练集
X = np.load("./XY_train/X.npy")
Y = np.load("./XY_train/Y.npy")
X_dev = np.load("./XY_dev/X_dev.npy")
Y_dev = np.load("./XY_dev/Y_dev.npy")