'''
Author: xmh
Date: 2021-06-08 20:20:42
LastEditors: xmh
LastEditTime: 2021-06-08 21:23:40
Description: 
FilePath: \DPCNN\modules\dpcnn.py
'''
import torch
import torch.nn as nn
from utils.config import config


class DPCNN(nn.Module):
    
    def __init__(self, pretrained_embedding=None):
        super().__init__()

        if config.using_pretrained_embedding:
            self.embeddings = nn.Embedding.from_pretrained_embedding(pretrained_embedding, freeze=False)
        else:
            self.embeddings = nn.Embedding(config.vocab_size, config.embedding_dim)
        
        self.region_embedding = nn.Conv2d(1, config.num_filter, (5, config.embedding_dim), stride=1)
        self.conv = nn.Conv2d(config.num_filter, config.num_filter, (5, 1), stride=1)
        self.max_pool = nn.MaxPool2d(kernel_size=(5, 1), stride=2)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(config.num_filter, )

    def forward(self, data_item):

        word_embeddings = self.embeddings(data_item['text'])
        

        


