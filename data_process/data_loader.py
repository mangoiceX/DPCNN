'''
Author: xmh
Date: 2021-06-08 10:46:11
LastEditors: xmh
LastEditTime: 2021-06-08 17:29:55
Description: 
FilePath: \DPCNN\data_process\data_loader.py
'''

from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from sklearn.model_selection import train_test_split


stop_words = stopwords.words('english')  # 得到停用词

with open('../data/corncob_lowercase.txt') as f:
    english_dictionary = []
    for line in f:
        line = line.rstrip()
        english_dictionary.append(line)
        
# print(stop_words)
# if "you'll" in stop_words:
#     print("exsit")
# else:
# 去除特殊符号


# 去除停用词
def remove_stop_words(text, wordnet_lemmatizer):
    # text = text.split(' ')
    # print(text)
    text = list(filter(lambda x: x not in stop_words, text))
    text = list(map(wordnet_lemmatizer.lemmatize, text))  # 词形统一
    # text = ''.join(text)

    return text
    
# 分词
# token_words = word_tokenize("it's a apple.")
# print(token_words)

def process_text(text:str, wordnet_lemmatizer)->list:
    # 去除特殊符号

    # 去除数字
    
    # 直接只保留英文单词（一步解决前面两个问题）
    text = text.split()
    # text = list(filter(wordnet.synsets, text))  # 这里太慢了
    text = list(filter(lambda x: x in english_dictionary, text))
    # 去除停用词
    # text = remove_stop_words(text, wordnet_lemmatizer)  # 删除之后，单词量大大减少，但是像like这种就不方便区分他只介词还是动词
    # 分词
    # token_words = word_tokenize(text)
    
    return text

def process_file(file_name:str):
    setences_list = []
    wordnet_lemmatizer = WordNetLemmatizer()
    cnt = 0
    with open(file_name, 'r', encoding='Windows-1252') as f:
        for line in f:
            line = line.rstrip()
            text = process_text(line, wordnet_lemmatizer)
            setences_list.append(text)
            cnt += 1
            if cnt > 20:
                break
    
    
    return setences_list

def process_data(file_name_pos, file_name_neg):
    
    setences_list_pos = process_file(file_name_pos)
    setences_list_neg = process_file(file_name_neg)

    # 添加标签
    # for i in range(len(setences_list_pos)):
    #     setences_list_pos[i].append(1)
    # for i in range(len(setences_list_neg)):
    #     setences_list_neg[i].append(0)
    # setences_list = setences_list_pos + setences_list_neg
    
    # labels = [1 for i in range(len(setences_list_pos))] + [0 for i in range(len(setences_list_neg))]
    
    # # 制作数据集
    # X_train, X_test, y_train, y_test = train_test_split(setences_list, labels, test_size=0.4, shuffle=True, random_state=0, stratify=labels)
    # X_test, X_valid, y_valid, y_valid = train_test_split(X_test, y_test, test_size=0.2, shuffle=True, random_state=0, stratify=labels)

    # return X_train, X_test, X_valid, y_train, y_valid, y_test

    return setences_list_pos, setences_list_neg


if __name__ == '__main__':
    file_name_pos = '../data/rt-polaritydata/rt-polarity.pos'
    file_name_neg = '../data/rt-polaritydata/rt-polarity.neg'
    file_name_pos_output = '../data/rt-polaritydata/rt-polarity_processed.pos'
    file_name_neg_output = '../data/rt-polaritydata/rt-polarity_processed.neg'
    setences_list_pos, setences_list_neg = process_data(file_name_pos, file_name_neg)
    
    print(setences_list_pos[:10])
    # with open(file_name_pos_output) as f:
    #     for li in setences_list_pos:
    #         f.write(li)
    #         f.write('\n')
    # with open(file_name_neg_output) as f:
    #     for li in setences_list_neg:
    #         f.write(li)
    #         f.write('\n')
    
    

    

    

    
        
