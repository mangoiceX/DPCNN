import os, sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # __file__获取执行文件相对路径，整行为取上一级的上一级目录
sys.path.append(BASE_DIR)

import torch
import torch.nn as nn
from utils.config import config
from tqdm import tqdm
from modules.dpcnn import DPCNN
import torch.nn.functional as F
from data_process.dataset_pytorch import ModelDataProcessor
import numpy as np


class Trainer:
    def __init__(self,
                train_data, 
                test_data
    ):
        self.train_data = train_data
        self.test_data = test_data

        self.model = DPCNN()
        self.init_network()
        # 设置优化器
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)
        # 学习率调控
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.5, patience=8, min_lr=1e-5, verbose=True)

        if config.USE_CUDA:
            self.model = self.model.cuda()
    
    def print_model(self):
        # for name, parameters in self.model.named_parameters():
        #     print(name, " : ", parameters.size())
        print("Total")
        self.get_parameters_number()
    
    def get_parameters_number(self):
        # 计算模型参数量大小
        total_num = sum(p.numel() for p in self.model.parameters())
        trainable_num = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(
            "Total number of parameters is {0}M, Trainable number is {1}M".format(total_num / 1e6, trainable_num / 1e6))
    
    

    def controller(self):
        self.print_model()

        for epoch in range(config.epochs):
            print("Epoch: {}".format(epoch))
            self.train(epoch)
            self.test(epoch)
    
    def get_loss(self, y_predict, y):
        
        loss = F.cross_entropy(y_predict, y)

        return loss
    
    def update(self, loss):
        self.optimizer.zero_grad()
        loss.backward()  # retain_graph=True
        self.optimizer.step()

    def train(self, epoch):
        print('STARTING TRAIN...')
        pbar = tqdm(enumerate(self.train_data), total=len(self.train_data))
        loss_total = 0.0
        correct_total = 0
        for i, data_item in pbar:
            y_predict = self.model(data_item['text_ids'])
            # y_predict = F.softmax(y_predict, dim=1)
            loss = self.get_loss(y_predict, data_item['label'])
            loss_total += float(loss)
            for i, j in zip(y_predict, data_item['label']):
                label_predict = i.argmax(dim=0)
                correct_total += (label_predict.cpu().numpy() == j.cpu().numpy())
            self.update(loss)
        loss_total_final = loss_total  # /len(self.train_data)/config.batch_size
        accuracy = correct_total/len(self.train_data)/config.batch_size
        print("train loss: {}, train accuracy: {}".format(loss_total_final, accuracy))
    
    def test(self, epoch):
        print('STARTING TESTING...')
        self.model.eval()
        pbar = tqdm(enumerate(self.test_data), total=len(self.test_data))
        loss_total = 0.0
        correct_total = 0
        for i, data_item in pbar:
            y_predict = self.model(data_item['text_ids'])
            y_predict = F.softmax(y_predict, dim=1)
            loss = self.get_loss(y_predict, data_item['label'])
            loss_total += float(loss)
            for i, j in zip(y_predict, data_item['label']):
                label_predict = i.argmax(dim=0)
                correct_total += (label_predict.cpu().numpy() == j.cpu().numpy())
        loss_total_final = loss_total  # /len(self.test_data)/config.batch_size
        accuracy = correct_total/len(self.test_data)/config.batch_size
        print("test loss: {}, test accuracy: {}".format(loss_total_final, accuracy))
    
    # 权重初始化，默认xavier
    def init_network(self, method='xavier', exclude='embeddings', seed=123):
        for name, w in self.model.named_parameters():
            if exclude not in name:
                if 'weight' in name:
                    if method == 'xavier':
                        nn.init.xavier_normal_(w)
                    elif method == 'kaiming':
                        nn.init.kaiming_normal_(w)
                    else:
                        nn.init.normal_(w)
                elif 'bias' in name:
                    nn.init.constant_(w, 0)
                else:
                    pass

if __name__ == "__main__":

    data_processor = ModelDataProcessor()
    train_data, test_data = data_processor.get_data_loader()

    trainer = Trainer(train_data, test_data)
    trainer.controller()

