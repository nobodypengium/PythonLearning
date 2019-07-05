import numpy as np
np.random.seed(0)
import keras
from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
np.random.seed(1)
from keras.initializers import glorot_uniform
from partVch2_1 import emo_utils

word_to_index, index_to_word, word_to_vec_map = emo_utils.read_glove_vecs('data/glove.6B.50d.txt')

def sentences_to_indices(X,word_to_index,max_len):
    """
    输入句子输出数字向量，每个元素是对应单词的index
    :param X:
    :param word_to_index:
    :param max_len:
    :return: X_indices
    """
    m = X.shape[0]
    X_indices = np.zeros((m,max_len))

    for i in range(m):
        sentences_words = X[i].lower().split()
        j=0
        for w in sentences_words:
            X_indices[i,j] = word_to_index[w]
            j+=1
    return X_indices

def pretrained_embedding_layer(word_to_vec_map,word_to_index):
    """
    创建keras的embedding层并加载训练好的权值
    :param word_to_vec_map:
    :param word_to_index:
    :return: embedding_layer: 一个训练好的keras层
    """
    vocab_len = len(word_to_index) + 1
    emb_dim = word_to_vec_map["cucumber"].shape[0]

    #初始化嵌入矩阵
    emb_matrix = np.zeros((vocab_len,emb_dim)) # 行存词，列存向量

    for word, index in word_to_index.items():
        emb_matrix[index,:] = word_to_vec_map[word]

    # 定义keras的embedding层并设置为不可训练（因为我们的数据集太小没有训练的必要）
    embedding_layer = Embedding(vocab_len,emb_dim,trainable=False) #输入一个一维的输出一个二维的
    embedding_layer.build((None,))
    embedding_layer.set_weights([emb_matrix])

    return  embedding_layer

def Emojify_V2(input_shape, word_to_vec_map, word_to_index):
    """
    实现Emojy-V2模型
    :param input_shape: 存储输入的形状(mzx_len,)
    :param word_to_vec_map:
    :param word_to_index:
    :return: model:返回用于生成emoji的模型
    """
    sentence_indices = Input(input_shape, dtype='int32')
    embedding_layer = pretrained_embedding_layer(word_to_vec_map,word_to_index)
    embeddings = embedding_layer(sentence_indices)
    X = LSTM(128,return_sequences=True)(embeddings)
    X = Dropout(0.5)(X)
    X = LSTM(128,return_sequences=False)(X)
    X = Dropout(0.5)(X)
    X=Activation('softmax')(X)

    model = Model(inputs=sentence_indices,outputs = X)
    return model