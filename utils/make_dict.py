'''
Author: xmh
Date: 2021-06-09 15:47:26
LastEditors: xmh
LastEditTime: 2021-06-09 16:22:27
Description: 构建词表
FilePath: \DPCNN\\utils\make_dict.py
'''


def make_dict(file_name_list, vocab_file):
    vocab_dict = {}
    cnt = 0
    for file_name in file_name_list:
        with open(file_name, 'r') as f:
            for line in f:
                line = line.rstrip().split()
                for word in line:
                    if word not in vocab_dict:
                        vocab_dict[word] = cnt
                        cnt += 1

    vocab_dict_sorted = sorted(vocab_dict.items(), key=lambda item: item[1], reverse=False)
    with open(vocab_file, 'w') as f:
        for item in vocab_dict_sorted:
            f.write(item[0])
            f.write('\n')
        
if __name__ == '__main__':
    file_name_pos = '../data/rt-polaritydata/rt-polarity_processed.pos'
    file_name_neg = '../data/rt-polaritydata/rt-polarity_processed.neg'
    vocab_file = '../data/vocab.txt'

    make_dict([file_name_pos, file_name_neg], vocab_file)




