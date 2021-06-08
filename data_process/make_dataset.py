'''
Author: xmh
Date: 2021-06-08 20:31:16
LastEditors: xmh
LastEditTime: 2021-06-08 20:37:42
Description: 
FilePath: \DPCNN\data_process\make_dataset.py
'''

from sklearn.model_selection import train_test_split


def process_file(file_name:str):
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


def process_data(file_name_pos, file_name_neg):
    setences_list_pos = process_file(file_name_pos)
    setences_list_neg = process_file(file_name_neg)

    # 添加标签
    for i in range(len(setences_list_pos)):
        setences_list_pos[i].append(1)
    for i in range(len(setences_list_neg)):
        setences_list_neg[i].append(0)
    setences_list = setences_list_pos + setences_list_neg
    
    labels = [1 for i in range(len(setences_list_pos))] + [0 for i in range(len(setences_list_neg))]
    
    # 制作数据集
    X_train, X_test, y_train, y_test = train_test_split(setences_list, labels, test_size=0.4, shuffle=True, random_state=0, stratify=labels)
    X_test, X_valid, y_valid, y_valid = train_test_split(X_test, y_test, test_size=0.2, shuffle=True, random_state=0, stratify=labels)

    return X_train, X_test, X_valid, y_train, y_valid, y_test


def get_data():
    # 提供给训练文件获取分割好的数据集
    file_name_pos = '../data/rt-polaritydata/rt-polarity_processed.pos'
    file_name_neg = '../data/rt-polaritydata/rt-polarity_processed.neg'
    X_train, X_test, X_valid, y_train, y_valid, y_test = process_data(file_name_pos, file_name_neg)

    return X_train, X_test, X_valid, y_train, y_valid, y_test

if __name__ == '__main__':
    X_train, X_test, X_valid, y_train, y_valid, y_test = get_data()
    print(X_train[:10])