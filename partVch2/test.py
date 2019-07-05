import numpy as np
from partVch2 import w2v_utils
from partVch2.netowrk import *

words,word_to_vec_map = w2v_utils.read_glove_vecs('data/glove.6B.50d.txt') #word:单词集合 ward_to_vec_map单词到嵌入向量的映射

g = word_to_vec_map['woman'] - word_to_vec_map['man']

#%% 测试发现有性别歧视
# print(g)
#
# name_list = ['john', 'marie', 'sophie', 'ronaldo', 'priya', 'rahul', 'danielle', 'reza', 'katy', 'yasmin']
#
# for w in name_list:
#     print (w, cosine_similarity(word_to_vec_map[w], g))
#
# word_list = ['lipstick', 'guns', 'science', 'arts', 'literature', 'warrior','doctor', 'tree', 'receptionist',
#              'technology',  'fashion', 'teacher', 'engineer', 'pilot', 'computer', 'singer']
# for w in word_list:
#     print (w, cosine_similarity(word_to_vec_map[w], g))

#%% 消除性别歧视
# e = "receptionist"
# print("去偏差前{0}与g的余弦相似度为：{1}".format(e, cosine_similarity(word_to_vec_map["receptionist"], g)))
#
# e_debiased = neutralize("receptionist", g, word_to_vec_map)
# print("去偏差后{0}与g的余弦相似度为：{1}".format(e, cosine_similarity(e_debiased, g)))

#%% 测试均衡算法，使在某个偏置上有区别的词相较于正交轴对称
print("==========均衡校正前==========")
print("cosine_similarity(word_to_vec_map[\"man\"], gender) = ", cosine_similarity(word_to_vec_map["man"], g))
print("cosine_similarity(word_to_vec_map[\"woman\"], gender) = ", cosine_similarity(word_to_vec_map["woman"], g))
e1, e2 = equalize(("man", "woman"), g, word_to_vec_map)
print("\n==========均衡校正后==========")
print("cosine_similarity(e1, gender) = ", cosine_similarity(e1, g))
print("cosine_similarity(e2, gender) = ", cosine_similarity(e2, g))