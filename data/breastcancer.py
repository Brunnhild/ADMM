import pandas as pd
import numpy as np


def get_data():
    #一共32个属性，删除了ID属性，并将肿瘤类型对应的文本属性换成了数字其中M--1，B--0，根据31个属性预测患乳腺癌的风险,
    train_data_all = np.array(pd.read_csv('BreastCancer/wdbc.csv'))
    train_data = np.array(train_data_all[:, 1:-1])  #属性
    train_score1 = np.array(train_data_all[:, -1])
    train_score = []
    for i in range(len(train_data)):
        if train_data[i][0] == 'M':
            train_data[i][0] = 1
        else:
            train_data[i][0] = 0
    for i in range(len(train_score1)):
        temp = []
        temp.append(train_score1[i])
        train_score.append(temp)
    train_score = np.array(train_score)  #风险结果
    train_score.reshape(-1, 1)
    return train_data, train_score
