import numpy as np
from partVch2 import w2v_utils
from partVch2.netowrk import *

words,word_to_vec_map = w2v_utils.read_glove_vecs('data/glove.6B.50d.txt') #word:单词集合 ward_to_vec_map单词到嵌入向量的映射
triads_to_try = [('italy', 'italian', 'spain'), ('india', 'delhi', 'japan'), ('man', 'woman', 'boy'), ('small', 'smaller', 'large')]
for triad in  triads_to_try:
    print ('{} -> {} <====> {} -> {}'.format( *triad, complete_analogy(*triad,word_to_vec_map)))