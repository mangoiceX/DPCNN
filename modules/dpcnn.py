'''
Author: xmh
Date: 2021-06-08 20:20:42
LastEditors: xmh
LastEditTime: 2021-06-09 12:06:19
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
        
        self.region_embedding = nn.Conv2d(1, config.num_filter, (3, config.embedding_dim), stride=1)
        self.conv = nn.Conv2d(config.num_filter, config.num_filter, (3, 1), stride=1)
        self.max_pool = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.act_fun = nn.ReLU()
        self.fc = nn.Linear(config.num_filter, config.num_rel)
        self.padding0 = nn.ZeroPad2d(0, 0, 1, 1)
        self.pooling = nn.MaxPool2d(kernel_size=(3, 1), stride=2)

    def forward(self, data_item):

        word_embeddings = self.embeddings(data_item['text'])  # [batch_size, seq_len, embedding_dim]
        region_word_embeddings = self.region_embedding(word_embeddings)  # [batch_size, seq_len-3+1, 1, num_filter]
        x = self.padding0(region_word_embeddings)  # [batch_size, seq_len, 1, num_filter]
        x = self.conv(self.act_fun(x))  # [batch_size, seq_len-3+1, 1, num_filter]
        x = self.padding0(x)  # [batch_size, seq_len, 1, num_filter]
        x = self.conv(self.act_fun(x))  # 
        x = x + region_word_embeddings  # 残差连接
        
        while x.size()[2] > 2:
            x = self._block(x)
        
        x = x.view(config.batch_size, 2*config.num_filter)  # [batch_size, 2, num_filter, 1]
        x = self.fc(x)
        
        return x
        
    def _block(self, x):

        px = self.pooling(x)  # [batch_size, (seq_len-2-2)/2, 1, num_filter]

        # 下面是两个等长卷积模块
        x = self.padding0(px) # 
        x = self.conv(self.act_fun(x))
        
        x = self.padding0(x)
        x = self.conv(self.act_fun(x))

        # 残差连接
        x = x + px
    
        return x
        


        
        


        
        
        

        


