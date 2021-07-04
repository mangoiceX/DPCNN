'''
Author: xmh
Date: 2021-06-08 14:06:39
LastEditors: xmh
LastEditTime: 2021-06-08 16:10:52
Description: 
FilePath: \DPCNN\test\gensim_test.py
'''
# from gensim.models import KeyedVectors
# wv_from_bin = KeyedVectors.load_word2vec_format('G:\GoogleNews-vectors-negative300.bin.gz', binary=True)  # , limit=200000
# print(wv_from_bin['.'].shape)
# print(wv_from_bin['having'].shape)
from gensim.models import FastText
from gensim.utils import tokenize
from gensim import utils
# from gensim.test.utils import datapath
import random

def read_sentence(filename):
    sentences_list = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.rstrip()
            line = line.split()
            sentences_list.append(line)

    return sentences_list

# class MyIter:
#     def __iter__(self):
#         # path = datapath('crime-and-punishment.txt')
#         path = '../'
#         with utils.open(path, 'r', encoding='utf-8') as fin:
#             for line in fin:
#                 yield list(tokenize(line))

class MyIter:
    def __init__(self,sentences_list):
        self.sentences_list = sentences_list

    def __iter__(self):
        for item in self.sentences_list:
            yield item

def make_dict(model):
    with open('../data/vocab_pretrain.txt', 'w') as f:
        for key in model.wv.key_to_index:
            f.write(key)
            f.write('\n')
    


if __name__ == '__main__':
    model = FastText(vector_size=128, window=3, min_count=1)  # instantiate
    pos_text = read_sentence('../data/rt-polaritydata/rt-polarity_processed.pos')
    neg_text = read_sentence('../data/rt-polaritydata/rt-polarity_processed.neg')
    sentences_list = pos_text + neg_text
    random.shuffle(sentences_list)
    print(len(sentences_list))
    model.build_vocab(corpus_iterable=sentences_list)  # MyIter(sentences)
    model.train(corpus_iterable=sentences_list, total_examples=len(sentences_list), epochs=10)  # train

    fname = './fasttext.model'
    model.save(fname)
    model = FastText.load(fname)
    # print(model.wv['silly'])
    # print(model.wv.key_to_index)
    make_dict(model)

