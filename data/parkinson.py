import pandas as pd
import numpy as np


def get_data():
    #26个属性，根据26个属性预测患帕金森症的风险
    train_data_all = np.array(pd.read_csv('Parkinson/parkinsons_updrs.csv'))
    train_data = np.array(train_data_all[:, :-1])  #属性
    train_score1 = np.array(train_data_all[:, -1])
    train_score = []
    for i in range(len(train_score1)):
        temp = []
        temp.append(train_score1[i])
        train_score.append(temp)
    train_score = np.array(train_score)  #风险结果
    train_score.reshape(-1, 1)
    return train_data, train_score
