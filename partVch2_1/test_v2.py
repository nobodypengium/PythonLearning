from partVch2_1.network_v2 import *

#%% 测试句子转index向量
# X1 = np.array(["funny lol", "lets play baseball", "food is ready for you"])
# X1_indices = sentences_to_indices(X1,word_to_index, max_len = 5)
# print("X1 =", X1)
# print("X1_indices =", X1_indices)

#%% 测试嵌入层
embedding_layer = pretrained_embedding_layer(word_to_vec_map,word_to_index)
print("weights[0][1][3] = ", embedding_layer.get_weights()[0][1][3])