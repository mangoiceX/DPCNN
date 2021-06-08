'''
Author: xmh
Date: 2021-06-08 20:45:03
LastEditors: xmh
LastEditTime: 2021-06-08 20:52:42
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
        
            