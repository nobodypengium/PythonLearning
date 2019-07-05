import numpy as np
from partVch2 import w2v_utils


def cosine_similarity(u, v):
    """
    用余弦相似度计算两个词嵌入向量的相似程度
    :param u: (n,)词向量
    :param v: (n,)词向量
    :return:cosine_similarity 相似度
    """
    distance = 0
    dot = np.dot(u, v)
    norm_u = np.sqrt(np.sum(np.power(u, 2)))
    norm_v = np.sqrt(np.sum(np.power(v, 2)))
    cosine_similarity = np.divide(dot, norm_u * norm_v)
    return cosine_similarity


def complete_analogy(word_a, word_b, word_c, word_to_vec_map):
    """
    解决A与B相比就像C与____相比的问题
    :param word_a: 字符串
    :param word_b: 字符串
    :param word_c: 字符串
    :param word_to_vec_map: 字典单词->Glove向量
    :return: best_word 满足v_b - v_a最接近v_best_word - v_c的词
    """
    word_a, word_b, word_c = word_a.lower(), word_b.lower(), word_c.lower()
    e_a, e_b, e_c = word_to_vec_map[word_a], word_to_vec_map[word_b], word_to_vec_map[word_c]
    words = word_to_vec_map.keys()

    # 初始化
    max_cosine_sim = -100
    best_word = None

    # 遍历数据集，找相似度最高的
    for word in words:
        # 避免匹配到输入的数据
        if word in [word_a, word_b, word_c]:
            continue
        cosine_sim = cosine_similarity((e_b - e_a), (word_to_vec_map[word] - e_c))

        if cosine_sim > max_cosine_sim:
            max_cosine_sim = cosine_sim
            best_word = word
    return best_word


def neutralize(word, g, word_to_vec_map):
    """
    将word投影到偏置轴上取得偏置轴投影的向量，用word减去这个向量就确保word没有该种偏置
    :param word: 将消除偏差的字符串
    :param g: 偏置轴
    :param word_to_vec_map: 嵌入向量
    :return: e_debiased 消除了偏差的嵌入向量
    """
    e = word_to_vec_map[word]
    e_biascomponent = np.divide(np.dot(e, g), np.square(np.linalg.norm(g))) * g
    e_debiased = e - e_biascomponent
    return e_debiased

def equalize(pair, bias_axis, word_to_vec_map):
    """
    消除给定偏差轴的偏差
    :param pair: 要消除偏差的词组
    :param bias_axis: 偏置轴
    :param word_to_vec_map: 嵌入向量
    :return:
    e_1:消除了偏差的第一个词
    e_2:消除了偏差的第二个词
    """
    w1, w2 = pair
    e_w1,e_w2 = word_to_vec_map[w1],word_to_vec_map[w2]
    mu = (e_w1+e_w2)/2.0
    mu_B = np.divide(np.dot(mu, bias_axis), np.square(np.linalg.norm(bias_axis))) * bias_axis #均值在偏置轴上的投影
    mu_orth = mu - mu_B #均值在正交轴上的投影
    e_w1B = np.divide(np.dot(e_w1, bias_axis), np.square(np.linalg.norm(bias_axis))) * bias_axis # w1在偏置轴上的投影
    e_w2B = np.divide(np.dot(e_w2, bias_axis), np.square(np.linalg.norm(bias_axis))) * bias_axis # w2在偏置轴上的投影
    corrected_e_w1B = np.sqrt(np.abs(1 - np.square(np.linalg.norm(mu_orth)))) * np.divide(e_w1B - mu_B,
                                                                                          np.abs(e_w1 - mu_orth - mu_B)) #调整w1的偏置部分
    corrected_e_w2B = np.sqrt(np.abs(1 - np.square(np.linalg.norm(mu_orth)))) * np.divide(e_w2B - mu_B,
                                                                                          np.abs(e_w2 - mu_orth - mu_B)) #调整w2的偏置部分
    e1 = corrected_e_w1B + mu_orth #调整后的偏置部分加上正交部分
    e2 = corrected_e_w2B + mu_orth

    return e1, e2

