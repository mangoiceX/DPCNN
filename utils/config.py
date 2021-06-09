'''
Author: xmh
Date: 2021-06-08 20:45:03
LastEditors: xmh
LastEditTime: 2021-06-09 12:06:53
Description: 
FilePath: \DPCNN\utils\config.py
'''

class Config:

    def __init__(self,
            embedding_dim=128,
        ):
        
        self.embedding_dim = embedding_dim
        self.using_pretrained_embedding = False
        self.vocab_size = 2000
        self.num_filter = 250
        self.num_rel = 2
        
        
            