# -*- coding:UTF-8 -*-

import numpy as np
import os
import time
import sys
import math
import gpu_train_feature as feature

# dataset_path_name = "./data/dataset/fix_dataset_0_30_1_10___20000101_20020101_20190414"
# dataset_merge_file_name = "./data/dataset/fix_dataset_0_30_1_10___20000101_20020101_20190414.npy"
dataset_daily_file_name = "./data/dataset/fix_dataset_daily_0_30_1_10___20000101_20120101_20190918.npy"

dataset_path_name = "./data/dataset/fix_dataset_0_30_1_10_0_0___20000101_20020101_20190414"
dataset_merge_file_name = "./data/dataset/fix_dataset_0_30_1_10_0_0___20000101_20020101_20190414.npy"

dataset_train_test_split_date = 20170101


def GetTrainTestDataMerge():
    if not os.path.exists(dataset_merge_file_name):
        init_flag = True
        file_count = 0
        for item in os.listdir(dataset_path_name):
            item_path = os.path.join(dataset_path_name, item)
            if os.path.isfile(item_path):
                if ".npy" in item:
                    stock_dataset = np.load(item_path)
                    if len(stock_dataset) > 0:
                        if init_flag:
                            init_flag = False
                            data_set = stock_dataset
                        else:
                            data_set = np.vstack((data_set, stock_dataset))
                        file_count += 1
                        print("%-4d : %s 100%%" % (file_count, item))
        np.save(dataset_merge_file_name, data_set)
    return np.load(dataset_merge_file_name)


def GetTrainTestDataSampleByDate(test_ratio):
    sample_num = int(1.0/test_ratio + 0.0001)
    dataset = GetTrainTestDataMerge()
    print("dataset: {}".format(dataset.shape))
    pos = ((dataset[:,feature.COL_TRADE_DATE(0)].astype(int) % 100) % sample_num) == 0
    test_data = dataset[pos]
    train_data = dataset[~pos]
    print("train: {}".format(train_data.shape))
    print("test: {}".format(test_data.shape))

    train_features = train_data[:, 0:feature.FEATURE_SIZE()]
    train_labels = train_data[:, feature.COL_ACTIVE_LABEL():feature.COL_ACTIVE_LABEL()+1]

    test_features = test_data[:, 0:feature.FEATURE_SIZE()]
    test_labels = test_data[:, feature.COL_ACTIVE_LABEL():feature.COL_ACTIVE_LABEL()+1]

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
    train_labels = train_data[:, feature.COL_ACTIVE_LABEL():feature.COL_ACTIVE_LABEL()+1]

    test_features = test_data[:, 0:feature.FEATURE_SIZE()]
    test_labels = test_data[:, feature.COL_ACTIVE_LABEL():feature.COL_ACTIVE_LABEL()+1]

    return train_features, train_labels, test_features, test_labels, test_data

def GetTrainTestDataSplitByDate():
    dataset = GetTrainTestDataMerge()

    # dataset=dataset[:20000]

    print("dataset: {}".format(dataset.shape))
    pos = dataset[:,feature.COL_TRADE_DATE(0)] < dataset_train_test_split_date
    train_data = dataset[pos]
    test_data = dataset[~pos]
    print("train: {}".format(train_data.shape))
    print("test: {}".format(test_data.shape))

    train_features = train_data[:, 0:feature.FEATURE_SIZE()]
    train_labels = train_data[:, feature.COL_ACTIVE_LABEL():feature.COL_ACTIVE_LABEL()+1]

    test_features = test_data[:, 0:feature.FEATURE_SIZE()]
    test_labels = test_data[:, feature.COL_ACTIVE_LABEL():feature.COL_ACTIVE_LABEL()+1]

    return train_features, train_labels, test_features, test_labels, test_data

def GetTestData():
    train_features, train_labels, test_features, test_labels, test_data = GetTrainTestDataSplitByDate()
    return test_data

def GetDailyDataSet(start_date):
    dataset = np.load(dataset_daily_file_name)
    pos = dataset[:,feature.COL_TRADE_DATE(0)] >= start_date
    dataset = dataset[pos]
    print("test_data: {}".format(dataset.shape))
    # raw_input("Enter ...")

    pos = dataset[:,feature.COL_TRADE_DATE(feature.active_label_day)] == feature.INVALID_DATE
    print("unfinished num: %u" % np.sum(pos))
    return dataset

if __name__ == "__main__":
    # 生成测试集和训练集数据
    dataset = GetTrainTestDataSplitByDate()
