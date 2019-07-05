import numpy as np
from partVch2_1 import emo_utils
import emoji
import matplotlib.pyplot as plt


def sentence_to_avg(sentence, word_to_vec_map):
    """
    将句子中的单词的嵌入向量平均
    :param sentence:
    :param word_to_vec_map:
    :return: avg:对句子的50维编码
    """
    words = sentence.lower().split()
    avg = np.zeros(50,)
    for w in words:
        avg+=word_to_vec_map[w]
    avg = np.divide(avg,len(words))
    return avg

def model(X,Y,word_to_vec_map,learning_rate=0.01,num_iterations=400):
    """
    训练词向量模型
    :param X: 输入字符串
    :param Y: 输出emoji标签
    :param word_to_vec_map: String -> 50维词向量
    :param learning_rate: 学习率
    :param num_iterations: 迭代次数
    :return:
    """
    np.random.seed(1)
    m = Y.shape[0]
    n_y = 5
    n_h = 50 #词向量长度

    #Xavier初始化参数
    W = np.random.randn(n_y,n_h)/np.sqrt(n_h)
    b = np.zeros((n_y,))

    Y_oh = emo_utils.convert_to_one_hot(Y,C=n_y)

    #优化
    for t in range(num_iterations):
        for i in range(m):
            avg = sentence_to_avg(X[i],word_to_vec_map)
            #前向传播
            z = np.dot(W,avg) + b
            a = emo_utils.softmax(z)
            #计算损失
            cost = -np.sum(Y_oh[i]*np.log(a))
            #计算梯度
            dz = a - Y_oh[i]
            dW = np.dot(dz.reshape(n_y,1),avg.reshape(1,n_h))
            db = dz
            #更新参数
            W = W - learning_rate*dW
            b = b - learning_rate*db
        if t % 100 == 0:
            print("第{t}轮，损失为{cost}".format(t=t,cost=cost))
            pred = emo_utils.predict(X,Y,W,b,word_to_vec_map) #最后一次学习完后的预测值

    return pred,W,b

