# -*- coding:UTF-8 -*-

import numpy as np
import os
import time
import sys
import math
import pandas as pd
import gpu_train_feature as feature

# dataset_file_name = "./data/dataset/wave_dataset_0_20020101_20000101_20000101_20190414___2_2_0_1_0_5_0.npy"
dataset_file_name = "./data/dataset/wave_dataset_0_30_0_0_20120101_20000101_20000101_20190414___2_2_0_1_0_5_0.npy"
# dataset_file_name = "./data/dataset/wave_dataset_0_30_0_0_20120101_20000101_20000101_20190414___2_2_0_1_0_1_0.npy"

dataset_train_test_split_date = 20180101


def COL_INCREASE():
    return feature.feature_size

def COL_TS_CODE():
    return feature.feature_size + 1

def COL_ON_PRETRADE_DATE():
    return feature.feature_size + 2

def COL_ON_DATE():
    return feature.feature_size + 3

def COL_OFF_DATE():
    return feature.feature_size + 4

def COL_HOLDING_DAYS():
    return feature.feature_size + 5

def GetTrainTestDataMerge():
    return np.load(dataset_file_name)

def SortWaveDataset(dataset):
    captions = []
    for iloop in range(0, feature.FEATURE_SIZE()):
        captions.append('f_%u' % iloop)
    captions.append('label')
    captions.append('ts_code')
    captions.append('pre_on_date')
    captions.append('on_date')
    captions.append('off_date')
    captions.append('holding_days')
    data_df = pd.DataFrame(dataset, columns=captions)
    data_df = data_df.sort_values(by=['pre_on_date', 'ts_code'], ascending=(True, True))
    return data_df.values

def GetTrainTestDataSampleByDate(test_ratio):
    sample_num = int(1.0/test_ratio + 0.0001)
    dataset = GetTrainTestDataMerge()
    print("dataset: {}".format(dataset.shape))
    pos = ((dataset[:,COL_ON_PRETRADE_DATE()].astype(int) % 100) % sample_num) == 0
    test_data = dataset[pos]
    train_data = dataset[~pos]
    print("train: {}".format(train_data.shape))
    print("test: {}".format(test_data.shape))

    test_data = SortWaveDataset(test_data)

    train_features = train_data[:, 0:feature.FEATURE_SIZE()]
    train_labels = train_data[:, feature.FEATURE_SIZE():feature.FEATURE_SIZE()+1]

    test_features = test_data[:, 0:feature.FEATURE_SIZE()]
    test_labels = test_data[:, feature.FEATURE_SIZE():feature.FEATURE_SIZE()+1]

    return train_features, train_labels, test_features, test_labels, test_data

def GetTrainTestDataRandom(test_ratio):
    sample_num = int(1.0/test_ratio + 0.0001)
    dataset = GetTrainTestDataMerge()
    print("dataset: {}".format(dataset.shape))
    print('sample_num:%u' % sample_num)
    # 生成数值范围在 0-（sample_num-1）的随机数组，pos是值为0的位置
    pos = (np.random.randint(0, sample_num, size=len(dataset)) == 0)
    test_data = dataset[pos]
    train_data = dataset[~pos]
    print("train: {}".format(train_data.shape))
    print("test: {}".format(test_data.shape))

    test_data = SortWaveDataset(test_data)

    train_features = train_data[:, 0:feature.FEATURE_SIZE()]
    train_labels = train_data[:, feature.FEATURE_SIZE():feature.FEATURE_SIZE()+1]

    test_features = test_data[:, 0:feature.FEATURE_SIZE()]
    test_labels = test_data[:, feature.FEATURE_SIZE():feature.FEATURE_SIZE()+1]

    return train_features, train_labels, test_features, test_labels, test_data

def GetTrainTestDataSplitByDate():
    dataset = GetTrainTestDataMerge()
    print("dataset: {}".format(dataset.shape))
    pos = dataset[:,COL_ON_PRETRADE_DATE()] < dataset_train_test_split_date
    train_data = dataset[pos]
    test_data = dataset[~pos]
    print("train: {}".format(train_data.shape))
    print("test: {}".format(test_data.shape))

    test_data = SortWaveDataset(test_data)

    train_features = train_data[:, 0:feature.FEATURE_SIZE()]
    train_labels = train_data[:, feature.FEATURE_SIZE():feature.FEATURE_SIZE()+1]

    test_features = test_data[:, 0:feature.FEATURE_SIZE()]
    test_labels = test_data[:, feature.FEATURE_SIZE():feature.FEATURE_SIZE()+1]

    return train_features, train_labels, test_features, test_labels, test_data

def GetTestData():
    train_features, train_labels, test_features, test_labels, test_data = GetTrainTestDataSplitByDate()
    return test_data

if __name__ == "__main__":
    # 生成测试集和训练集数据
    dataset = GetTrainTestDataSplitByDate()
