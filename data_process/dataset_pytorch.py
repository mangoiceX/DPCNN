'''
Author: xmh
Date: 2021-06-09 13:11:50
LastEditors: xmh
LastEditTime: 2021-06-09 15:29:16
Description:
FilePath: \DPCNN\\data_process\\dataset_pytorch.py
'''

import torch
from utils.config import config
from sklearn.model_selection import train_test_split
import copy


class ModelDataProcessor:
    def __init__(self):
        self.get_dict()

    def get_dict(self):
        self.word_dict = {}
        with open(config.vocab_file, 'r') as f:
            cnt = 0
            for line in f:
                line = line.rstrip()
                self.word_dict[line] = cnt
                cnt += 1

    def process_file(self, file_name:str):
        setences_list = []
        # cnt = 0
        with open(file_name, 'r', encoding='Windows-1252') as f:
            for line in f:
                text = line.rstrip().split()
                setences_list.append(text)
                # cnt += 1
                # if cnt > 20:
                #     break

        return setences_list


    def process_data(self, file_name_pos, file_name_neg):
        setences_list_pos = self.process_file(file_name_pos)
        setences_list_neg = self.process_file(file_name_neg)

        # # 添加标签
        # for i in range(len(setences_list_pos)):
        #     setences_list_pos[i].append(1)
        # for i in range(len(setences_list_neg)):
        #     setences_list_neg[i].append(0)
        setences_list = setences_list_pos + setences_list_neg
        
        labels = [1 for i in range(len(setences_list_pos))] + [0 for i in range(len(setences_list_neg))]
        
        # 制作数据集
        X_train, X_test, y_train, y_test = train_test_split(setences_list, labels, test_size=0.3, shuffle=True, random_state=0, stratify=labels)
        # X_test, X_valid, y_valid, y_valid = train_test_split(X_test, y_test, test_size=0.2, shuffle=True, random_state=0, stratify=labels)

        # return X_train, X_test, X_valid, y_train, y_valid, y_test
        return X_train, X_test, y_train, y_test

    def get_data(self):
        # 提供给训练文件获取分割好的数据集
        file_name_pos = '../data/rt-polaritydata/rt-polarity_processed.pos'
        file_name_neg = '../data/rt-polaritydata/rt-polarity_processed.neg'
        # X_train, X_test, X_valid, y_train, y_valid, y_test = self.process_data(file_name_pos, file_name_neg)
        X_train, X_test, y_train, y_test = self.process_data(file_name_pos, file_name_neg)

        # return X_train, X_test, X_valid, y_train, y_valid, y_test
        return X_train, X_test, y_train, y_test

    def get_data_loader(self):
       
        # X_train, X_test, X_valid, y_train, y_valid, y_test = self.get_data()
        X_train, X_test, y_train, y_test = self.get_data()
        # 中间应该还增加对文本的编码
        train_text_ids = [[self.word_dict[word] for word in item] for item in X_train]
        test_text_ids = [[self.word_dict[word] for word in item] for item in X_test]
    
        train_data = {'text_origin': X_train, 'label': y_train, 'text_ids': train_text_ids}
        # valid_data = {'text': X_valid, 'label': y_valid}
        test_data = {'text_origin': X_test, 'label': y_test, 'text_ids': test_text_ids}
        
        train_data = DataSet(train_data)
        # valid_data = DataSet(valid_data)
        test_data = DataSet(test_data)
        
        train_loader = torch.utils.data.DataLoader(
            dataset=train_data,
            batch_size=config.batch_size,
            collate_fn=train_data.collate_fn,
            shuffle=False,
            drop_last=True
        )

        test_loader = torch.utils.data.DataLoader(
            dataset=test_data,
            batch_size=config.batch_size,
            collate_fn=test_data.collate_fn,
            shuffle=False,
            drop_last=True
        )

        return train_loader, test_loader


class DataSet(torch.utils.data.Dataset):
    
    def __init__(self, data):
        self.data = copy.deepcopy(data)
    
    def __getitem__(self, index):
        text_origin = self.data['text_origin'][index]
        label = self.data['label'][index]
        text_ids = self.data['text_ids'][index]
        
        data_info = {}
        for key in self.data.keys():
            if key in locals():
                data_info[key] = locals()[key]
        
        return data_info

    def __len__(self):
        return len(self.data['text_origin'])
    
    def collate_fn(self, data_batch):

        def merge(sequences):
            lengths = [len(seq) for seq in sequences]
            max_length = max(lengths)
            seq_padded = torch.zeros(len(sequences), max_length)
            masked_tokens = torch.zeros(len(sequences), max_length)
            tmp_padded = torch.ones(1, max_length)
            
            for i, seq in enumerate(sequences):
                end = lengths[i]
                seq = torch.LongTensor(seq)  # 转化为整数
                if len(seq) != 0:
                    seq_padded[i, :end] = seq[:end]
                    masked_tokens[i, :end] = tmp_padded[0, :end]
            
            return seq_padded, masked_tokens
        
        item_info = {}  # 对数据按照特征进行聚合
        # print(data_batch)
        for key in data_batch[0].keys():
            item_info[key] = [d[key] for d in data_batch]
        
        text_ids, mask_ids = merge(item_info['text_ids'])
        label = torch.Tensor(item_info['label']).to(dtype=torch.int64)
        if config.USE_CUDA:
            text_ids = text_ids.contiguous().cuda()
            mask_ids = mask_ids.contiguous().cuda()
            label = label.contiguous().cuda()
        else:        
            text_ids = text_ids.contiguous()
            mask_ids = mask_ids.contiguous()
            label = label.contiguous()
        data_info = {'text_origin': item_info['text_origin']}
        for key in item_info.keys():
            if key in locals():
                data_info[key] = locals()[key]
        
        return data_info


        
            
    

