'''
Author: xmh
Date: 2021-06-08 20:20:42
LastEditors: xmh
LastEditTime: 2021-06-09 12:09:42
Description: 
FilePath: \DPCNN\modules\dpcnn.py
'''
import torch
import torch.nn as nn
from utils.config import config
# torch.backends.cudnn.enabled = False


class DPCNN(nn.Module):
    
    def __init__(self, pretrained_embedding=None):
        super().__init__()

        # if config.using_pretrained_embedding:
        if pretrained_embedding is not None:
            print("using pretrained_embedding")
            self.embeddings = nn.Embedding.from_pretrained(torch.FloatTensor(pretrained_embedding), freeze=False)
        else:
            self.embeddings = nn.Embedding(config.vocab_size, config.embedding_dim)
        
        self.region_embedding = nn.Conv2d(1, config.num_filter, (3, config.embedding_dim), stride=1)
        self.conv = nn.Conv2d(config.num_filter, config.num_filter, (3, 1), stride=1)
        self.max_pool = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.act_fun = nn.ReLU()
        self.fc = nn.Linear(config.num_filter, config.num_rel)
        self.padding0 = nn.ZeroPad2d((0, 0, 1, 1))
        self.padding1 = nn.ZeroPad2d((0, 0, 0, 1))  # bottom
        self.pooling = nn.MaxPool2d(kernel_size=(3, 1), stride=2)

    def forward(self, data_item):

        word_embeddings = self.embeddings(data_item.to(torch.int64))  # [batch_size, seq_len, embedding_dim]
        word_embeddings = word_embeddings.unsqueeze(1)  # [batch_size, 1, seq_len, embedding_dim]
        region_word_embeddings = self.region_embedding(word_embeddings)  # [batch_size, num_filter, seq_len-3+1, 1]
        x = self.padding0(region_word_embeddings)  # [batch_size, num_filter, seq_len, 1]
        x = self.conv(self.act_fun(x))  # [batch_size, num_filter, seq_len-3+1, 1]
        x = self.padding0(x)  # [batch_size, num_filter, seq_len, 1]
        x = self.conv(self.act_fun(x))  # [batch_size, num_filter, seq_len-3+1, 1]
        x = x + region_word_embeddings  # 残差连接
        
        while x.size()[2] >= 2:  # 直到的seq_len数量减少到1
            x = self._block(x)
        x = x.squeeze()  # [batch_size, num_filter, 1, 1] -> [batch_size, num_filters]
        # print(x.shape)
        # x = x.view(config.batch_size, config.num_filter)  # [batch_size, num_class_num]
        x = self.fc(x)
        
        return x
        
    def _block(self, x):

        x = self.padding1(x)
        px = self.pooling(x)  # [batch_size, (seq_len-2-2)/2, 1, num_filter]

        # 下面是两个等长卷积模块
        x = self.padding0(px) # 
        x = self.conv(self.act_fun(x))
        
        x = self.padding0(x)
        # print(x)
        x = self.conv(self.act_fun(x))

        # 残差连接
        x = x + px
    
        return x
        


        
        


        
        
        

        


