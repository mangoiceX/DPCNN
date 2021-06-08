'''
Author: xmh
Date: 2021-06-08 14:06:39
LastEditors: xmh
LastEditTime: 2021-06-08 16:10:52
Description: 
FilePath: \DPCNN\test\gensim_test.py
'''
from gensim.models import KeyedVectors
wv_from_bin = KeyedVectors.load_word2vec_format('G:\GoogleNews-vectors-negative300.bin.gz', binary=True)  # , limit=200000
print(wv_from_bin['.'].shape)
print(wv_from_bin['having'].shape)


