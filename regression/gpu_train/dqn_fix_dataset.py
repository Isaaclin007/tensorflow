# -*- coding:UTF-8 -*-


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
import datetime
import sys
import math

import_tushare_data = os.path.exists('tushare_data.py')
if import_tushare_data:
    import tushare_data

import_feature = os.path.exists('feature.py')
if import_feature:
    import feature

import_pp_daily_update = os.path.exists('pp_daily_update.py')
if import_pp_daily_update:
    import pp_daily_update

import_dqn_dataset = os.path.exists('dqn_dataset.py')
if import_dqn_dataset:
    import dqn_dataset

import_np_common = os.path.exists('../common/np_common.py')
if import_np_common:
    sys.path.append("..")
    from common import np_common

label_days = 10
decay_ratio = 0.8


def SettingName():
    temp_name = '%s_%s_%u_%u_%f' % (dqn_dataset.SettingName(), 
                            tushare_data.train_test_date, 
                            dqn_dataset.dataset_train_test_split_date, 
                            label_days, 
                            decay_ratio)
    return temp_name

if import_tushare_data and import_dqn_dataset:
    setting_name_ = SettingName()
    dqn_dataset_file_name_ = dqn_dataset.FileNameDataSet(False)
    dqn_dataset_ACTURE_DATA_INDEX_DATE_ = dqn_dataset.ACTURE_DATA_INDEX_DATE()
    dqn_dataset_ACTURE_DATA_INDEX_OPEN_ = dqn_dataset.ACTURE_DATA_INDEX_OPEN()
    dqn_dataset_ACTURE_DATA_INDEX_TSCODE_ = dqn_dataset.ACTURE_DATA_INDEX_TSCODE()
    feature_FEATURE_SIZE_ = feature.FEATURE_SIZE()
    dqn_dataset_dataset_train_test_split_date_ = dqn_dataset.dataset_train_test_split_date
else:
    dqn_dataset_ACTURE_DATA_INDEX_DATE_ = 156
    dqn_dataset_ACTURE_DATA_INDEX_OPEN_ = 152
    dqn_dataset_ACTURE_DATA_INDEX_TSCODE_ = 155
    feature_FEATURE_SIZE_ = 150
    dqn_dataset_dataset_train_test_split_date_ = 20170101
    setting_name_ = '0_30_1_10_0_0___20000101_10_20020101_20190414_%u_%u_%.6f' % (
        dqn_dataset_dataset_train_test_split_date_, 
        label_days, 
        decay_ratio)
    dqn_dataset_file_name_ = './data/dataset/dqn_0_30_1_10_0_0___20000101_10_20020101_20190414.npy'


def FileNameDataSet():
    file_name = './data/dataset/dqn_fix_%s.npy' % setting_name_
    return file_name

def CreateDataSet():
    dataset_file_name = FileNameDataSet()
    if os.path.exists(dataset_file_name):
        print('dataset already exist: %s' % dataset_file_name)
        return

    dqn_dataset_file_name = dqn_dataset_file_name_
    dqn_src_dataset = np.load(dqn_dataset_file_name)
    print("dqn_src_dataset: {}".format(dqn_src_dataset.shape))

    date_num = dqn_src_dataset.shape[0]
    code_num = dqn_src_dataset.shape[1]
    data_unit_date_index = dqn_dataset_ACTURE_DATA_INDEX_DATE_
    data_unit_open_index = dqn_dataset_ACTURE_DATA_INDEX_OPEN_
    data_unit_tscode_index = dqn_dataset_ACTURE_DATA_INDEX_TSCODE_
    feature_size = feature_FEATURE_SIZE_

    # feature | label(从feature_date+1开始计算) | feature_date | ts_code
    dataset = np.zeros((date_num * code_num, feature_size + 3))
    data_num = 0
    max_label_date_num = label_days * 2
    
    for code_index in range(0, code_num):
        stock_code = dqn_src_dataset[0][code_index][data_unit_tscode_index]
        for day_loop in reversed(range(0, date_num)):
            temp_date = dqn_src_dataset[day_loop][code_index][data_unit_date_index]
            # if temp_date > dqn_dataset.dataset_train_test_split_date:
            #     break
            if (temp_date > 0):
                temp_price_ratio = 1.0
                temp_count = 0
                temp_date_count = 0
                temp_index = day_loop - 1
                current_price = dqn_src_dataset[day_loop][code_index][data_unit_open_index]
                temp_effect_ratio = 1.0
                while(temp_index >= 0):
                    if dqn_src_dataset[temp_index][code_index][data_unit_date_index] > 0:
                        temp_price = dqn_src_dataset[temp_index][code_index][data_unit_open_index]
                        temp_increase = temp_price / current_price - 1.0
                        decay_increase = temp_increase * temp_effect_ratio
                        temp_price_ratio *= (1.0 + decay_increase)
                        temp_effect_ratio *= decay_ratio
                        temp_count += 1
                        current_price = temp_price
                        if temp_count == label_days:
                            break
                    temp_index -= 1
                    temp_date_count += 1
                    if temp_date_count > max_label_date_num:
                        break
                if temp_count == label_days:
                    temp_label = (temp_price_ratio - 1.0) * 100.0
                    dataset[data_num][:feature_size] = dqn_src_dataset[day_loop][code_index][:feature_size]
                    dataset[data_num][feature_size] = temp_label
                    dataset[data_num][feature_size + 1] = temp_date
                    dataset[data_num][feature_size + 2] = stock_code
                    data_num += 1
        print("%-4d : %06.0f 100%%" % (code_index, stock_code))
    dataset = dataset[:data_num]

    print("dataset: {}".format(dataset.shape))
    print("file_name: %s" % dataset_file_name)
    np.save(dataset_file_name, dataset)

def ShowDataSet(dataset, caption):
    for iloop in range(len(dataset)):
        print("\n%s[%u]:" % (caption, iloop))
        print("-" * 80),
        for dloop in range(dataset.shape[1]):
            if dloop % 5 == 0:
                print("")
            print("%-16.4f" % dataset[iloop][dloop]),
        print("")
    print("\n")

def GetDataSet():
    CreateDataSet()
    dataset_file_name = FileNameDataSet()
    dataset = np.load(dataset_file_name)
    print("dataset: {}".format(dataset.shape))
    #################### show random sample train data ##############
    # ShowDataSet(np_common.RandSelect(dataset, 10), 'rand_dataset')
    # rand_array = np.arange(dataset.shape[0])
    # np.random.shuffle(rand_array)
    # rand_dataset = dataset[rand_array[0:10]]
    # for iloop in range(len(rand_dataset)):
    #     print("\nrand_dataset[%u]:" % iloop),
    #     for dloop in range(dataset.shape[1]):
    #         if dloop % 5 == 0:
    #             print("")
    #         print("%-16.4f" % rand_dataset[iloop][dloop]),
    #     print("")
    #     # print(rand_dataset[iloop][:feature.FEATURE_SIZE()])
    #     # print(rand_dataset[iloop][feature.FEATURE_SIZE():])
    # print("\n")
    ##################################################################

    #################### 还需要显示前10000的label随机抽取N条的数据 ##############
    # sort_dataset = np_common.Sort2D(dataset, [feature.FEATURE_SIZE()], False)
    # ShowDataSet(np_common.RandSelect(sort_dataset[:10], 10), 'max10000_rand_dataset')

    #################### 还需要显示最后10000的label随机抽取N条的数据 ##############
    # sort_dataset = np_common.Sort2D(dataset, [feature.FEATURE_SIZE()], True)
    # ShowDataSet(np_common.RandSelect(sort_dataset[:10], 10), 'min10000_rand_dataset')

    #################### 显示 label 直方图 ##############
    # labels = dataset[:, feature.FEATURE_SIZE()]
    # np_common.ShowHist(labels)
    
    ###################################################################################

    pos = dataset[:,feature_FEATURE_SIZE_+1] < dqn_dataset_dataset_train_test_split_date_
    train_data = dataset[pos]
    test_data = dataset[~pos]
    print("train: {}".format(train_data.shape))
    print("test: {}".format(test_data.shape))
    train_features = train_data[:, 0:feature_FEATURE_SIZE_].astype(np.float32)
    train_labels = train_data[:, feature_FEATURE_SIZE_:feature_FEATURE_SIZE_+1].astype(np.float32)
    val_features = test_data[:, 0:feature_FEATURE_SIZE_].astype(np.float32)
    val_labels = test_data[:, feature_FEATURE_SIZE_:feature_FEATURE_SIZE_+1].astype(np.float32)
    return train_features, train_labels, val_features, val_labels


if __name__ == "__main__":
    train_features, train_labels, val_features, val_labels = GetDataSet()
    print("train_features: {}".format(train_features.shape))
    print("train_labels: {}".format(train_labels.shape))
    print("val_features: {}".format(val_features.shape))
    print("val_labels: {}".format(val_labels.shape))
    # for iloop in range(0, 10):
    #     print(train_features[iloop * 10000])
    #     print(train_labels[iloop * 10000])


