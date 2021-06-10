'''
Author: xmh
Date: 2021-06-08 20:45:03
LastEditors: xmh
LastEditTime: 2021-06-09 12:06:53
Description:
FilePath: \DPCNN\\utils\\config.py
'''

import torch

class Config:

    def __init__(self,
            embedding_dim=128,
        ):
        
        self.embedding_dim = embedding_dim
        self.using_pretrained_embedding = False
        self.num_filter = 250
        self.num_rel = 2
        self.batch_size = 4
        self.vocab_file = '../data/vocab.txt'

        cnt = 1  # 添加padd的位置
        with open(self.vocab_file, 'r') as f:
            for line in f:
                cnt += 1
        self.vocab_size = cnt
        self.epochs = 10
        self.lr = 0.001

        if torch.cuda.is_available():
            self.USE_CUDA = True
        else:
            self.USE_CUDA = False


config = Config()

