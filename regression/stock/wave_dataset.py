# -*- coding:UTF-8 -*-

import numpy as np
import os
import time
import sys
import math
import pandas as pd
import feature
import wave_kernel

dataset_train_test_split_date = 20180101


def GetTrainTestDataMerge():
    return np.load(wave_kernel.FileNameDataSet())


def GetTrainTestDataSampleByDate(test_ratio):
    sample_num = int(1.0/test_ratio + 0.0001)
    dataset = GetTrainTestDataMerge()
    print("dataset: {}".format(dataset.shape))
    pos = ((dataset[:,wave_kernel.COL_ON_PRETRADE_DATE()].astype(int) % 100) % sample_num) == 0
    test_data = dataset[pos]
    train_data = dataset[~pos]
    print("train: {}".format(train_data.shape))
    print("test: {}".format(test_data.shape))

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

    train_features = train_data[:, 0:feature.FEATURE_SIZE()]
    train_labels = train_data[:, feature.FEATURE_SIZE():feature.FEATURE_SIZE()+1]

    test_features = test_data[:, 0:feature.FEATURE_SIZE()]
    test_labels = test_data[:, feature.FEATURE_SIZE():feature.FEATURE_SIZE()+1]

    return train_features, train_labels, test_features, test_labels, test_data

def SortWaveDataset(dataset):
    captions = []
    for iloop in range(0, feature.FEATURE_SIZE()):
        captions.append('f_%u' % iloop)
    captions.append('label')
    captions.append('ts_code')
    captions.append('pre_on_date')
    captions.append('on_date')
    captions.append('pre_off_date')
    captions.append('off_date')
    captions.append('holding_days')
    data_df = pd.DataFrame(dataset, columns=captions)
    data_df = data_df.sort_values(by=['pre_on_date', 'ts_code'], ascending=(True, True))
    return data_df.values

def GetTrainTestDataSplitByDate():
    dataset = GetTrainTestDataMerge()
    print("dataset: {}".format(dataset.shape))
    pos = dataset[:,wave_kernel.COL_ON_PRETRADE_DATE()] < dataset_train_test_split_date
    train_data = dataset[pos]
    test_data = dataset[~pos]

    test_data = SortWaveDataset(test_data)

    print("train: {}".format(train_data.shape))
    print("test: {}".format(test_data.shape))

    train_features = train_data[:, 0:feature.FEATURE_SIZE()]
    train_labels = train_data[:, feature.FEATURE_SIZE():feature.FEATURE_SIZE()+1]

    test_features = test_data[:, 0:feature.FEATURE_SIZE()]
    test_labels = test_data[:, feature.FEATURE_SIZE():feature.FEATURE_SIZE()+1]

    return train_features, train_labels, test_features, test_labels, test_data

def GetTestData():
    train_features, train_labels, test_features, test_labels, test_data = GetTrainTestDataSplitByDate()
    return test_data

def GetDailyDataSet():
    data_set = np.load(wave_kernel.FileNameDailyDataSet())
    print("data_set: {}".format(data_set.shape))
    data_set = data_set[np.where(data_set[:,wave_kernel.COL_ON_PRETRADE_DATE()] >= 20190415)]
    print("data_set split: {}".format(data_set.shape))
    # debug_df = data_df[(data_df['pre_on_date'] == 20190327.0) & (data_df['ts_code'] == 002605.0)]
    # print(debug_df)
    return SortWaveDataset(data_set)

if __name__ == "__main__":
    # 生成测试集和训练集数据
    dataset = GetTrainTestDataSplitByDate()
