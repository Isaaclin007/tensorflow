# -*- coding:UTF-8 -*-


import tushare as ts
import numpy as np
import pandas as pd
import os
import time
import datetime
import sys
import math
import tushare_data
import feature
import pp_daily_update

dataset_start_date = 20120101
dataset_train_test_split_date = 20200101

def SettingName():
    temp_name = '%u_%u_%u_%u_%s_%s_%s_%u' % ( \
        feature.feature_type, \
        feature.feature_days, \
        feature.label_type, \
        feature.label_days, \
        tushare_data.industry_filter, \
        tushare_data.code_filter, \
        tushare_data.stocks_list_end_date, \
        dataset_start_date)
    return temp_name

def TrainSettingName():
    temp_name = '%s_%u' % (SettingName(), dataset_train_test_split_date)
    return temp_name

def FileNameFixDataSet():
    file_name = './data/dataset/fix_dataset_%s_%s.npy' % ( \
        SettingName(), \
        tushare_data.train_test_date)
    return file_name

def FileNameFixDataSetDaily():
    file_name = './data/dataset/fix_dataset_daily_%s_%s.npy' % ( \
        SettingName(), \
        pp_daily_update.update_date)
    return file_name

def AppendFeature(pp_data, day_index, data_unit):
    if feature_type == FEATURE_G0_D5_AVG:
        base_close = pp_data['close_100_avg'][day_index]
        base_vol = pp_data['vol_100_avg'][day_index]
        for iloop in reversed(range(0, feature_days)):
            temp_index = day_index + iloop
            data_unit.append(pp_data['open'][temp_index] / base_close)
            data_unit.append(pp_data['close'][temp_index] / base_close)
            data_unit.append(pp_data['high'][temp_index] / base_close)
            data_unit.append(pp_data['low'][temp_index] / base_close)
            data_unit.append(pp_data['vol'][temp_index] / base_vol)
    return True

def UpdateFixDataSet(is_daily_data, save_unfinished_record):
    if is_daily_data:
        dataset_file_name = FileNameFixDataSetDaily()
        pp_merge_data = pp_daily_update.GetPreprocessedMergeData()
    else:
        dataset_file_name = FileNameFixDataSet()
    if os.path.exists(dataset_file_name):
        print('dataset already exist: %s' % dataset_file_name)
        return
    code_list = tushare_data.StockCodes()
    init_flag = True
    for code_index in range(0, len(code_list)):
        stock_code = code_list[code_index]
        if is_daily_data:
            pp_data = pp_daily_update.GetPreprocessedData(pp_merge_data, stock_code)
        else:
            stock_pp_file_name = tushare_data.FileNameStockPreprocessedData(stock_code)
            if os.path.exists(stock_pp_file_name):
                pp_data = pd.read_csv(stock_pp_file_name)
            else:
                pp_data = []
        if len(pp_data) > 0:
            pp_data = pp_data[pp_data['trade_date'] >= int(dataset_start_date)].copy().reset_index(drop=True)
            data_list = []
            if save_unfinished_record:
                valid_data_num = len(pp_data) - feature.feature_days
                start_index = 0
            else:
                valid_data_num = len(pp_data) - feature.feature_days - feature.label_days
                start_index = feature.label_days
            if valid_data_num > 0:
                for day_loop in range(start_index, start_index + valid_data_num):
                    data_unit = feature.GetDataUnit(pp_data, day_loop)
                    if len(data_unit) > 0:
                        data_list.append(data_unit)
                temp_np_data = np.array(data_list)
                if init_flag:
                    data_set = temp_np_data
                    init_flag = False
                else:
                    data_set = np.vstack((data_set, temp_np_data))
            print("%-4d : %s 100%%" % (code_index, stock_code))
            # print("train_data: {}".format(train_data.shape))
            # print(train_data)
        # if (code_index > 0) and ((code_index % 100) == 0):
        #     print("dataset: {}".format(data_set.shape))
        #     np.save(dataset_file_name, data_set)
    print("dataset: {}".format(data_set.shape))
    np.save(dataset_file_name, data_set)

def CheckFixDataSet():
    dataset = np.load(FileNameFixDataSet())
    print("dataset: {}".format(dataset.shape))
    for iloop in range(0, feature.label_days):
        pos = dataset[:,feature.COL_TRADE_DATE(iloop)] == feature.INVALID_DATE
        if np.sum(pos) > 0:
            print('CheckFixDataSet.ERROR')
            print(dataset[pos])

def GetTrainDataTempSave():
    dataset = np.load(FileNameFixDataSet())
    print("dataset: {}".format(dataset.shape))

    pos = dataset[:,feature.COL_TRADE_DATE(0)] < dataset_train_test_split_date
    train_data = dataset[pos]
    print("train_data: {}".format(train_data.shape))

    features = train_data[:,0:feature.FEATURE_SIZE()]
    labels = train_data[:,feature.COL_ACTIVE_LABEL():feature.COL_ACTIVE_LABEL()+1]

    np.save('./data/dataset/fix_dataset_temp_features.npy', features)
    np.save('./data/dataset/fix_dataset_temp_labels.npy', labels)

def GetTrainData():
    GetTrainDataTempSave()
    features = np.load('./data/dataset/fix_dataset_temp_features.npy')
    labels = np.load('./data/dataset/fix_dataset_temp_labels.npy')
    train_data = np.append(features, labels, axis=1)
    print("train_data: {}".format(train_data.shape))
    # raw_input("Enter ...")

    print("reorder...")
    order=np.argsort(np.random.random(len(train_data)))
    train_data=train_data[order]
    train_data=train_data[:2000000]
    # raw_input("Enter ...")
    # sample_train_data = train_data[:10]

    print("get feature ...")
    train_features = train_data[:, 0:feature.FEATURE_SIZE()].copy()
    # raw_input("Enter ...")

    print("get label...")
    train_labels = train_data[:, feature.FEATURE_SIZE():feature.FEATURE_SIZE()+1].copy()
    # raw_input("Enter ...")
    print("train_features: {}".format(train_features.shape))
    print("train_labels: {}".format(train_labels.shape))
    return train_features, train_labels

# def GetTrainData():
#     dataset = np.load(FileNameFixDataSet())
#     print("dataset: {}".format(dataset.shape))
#     # raw_input("Enter ...")

#     pos = (dataset[:,feature.COL_TRADE_DATE(0)] < dataset_train_test_split_date)
#     print(dataset[:,feature.COL_TRADE_DATE(0)])
#     print(pos)
#     train_data = dataset[pos]
#     print("train_data: {}".format(train_data.shape))

#     print("reorder...")
#     order=np.argsort(np.random.random(len(train_data)))
#     train_data=train_data[order]
#     train_data=train_data[:2000000]
#     # raw_input("Enter ...")
#     # sample_train_data = train_data[:10]

#     label_index = feature.COL_ACTIVE_LABEL()
#     print("get feature ...")
#     train_features = train_data[:, 0:feature_size].copy()
#     # raw_input("Enter ...")

#     print("get label...")
#     train_labels = train_data[:, label_index:label_index+1].copy()
#     # raw_input("Enter ...")
#     print("train_features: {}".format(train_features.shape))
#     print("train_labels: {}".format(train_labels.shape))

#     # caption = GetTrainDataCaption()
#     # print('caption[%u]:' % len(caption))
#     # print(caption)
    
#     # sample_train_data_df = pd.DataFrame(sample_train_data, columns=caption)
#     # sample_train_data_df.to_csv('./sample_train_data_df.csv')
#     return train_features, train_labels

def GetTestData():
    dataset = np.load(FileNameFixDataSet())
    print("dataset: {}".format(dataset.shape))
    # raw_input("Enter ...")

    pos = dataset[:,feature.COL_TRADE_DATE(0)] >= dataset_train_test_split_date
    # pos = dataset[:,feature.COL_TRADE_DATE(0)] >= 20170101
    test_data = dataset[pos]
    print("test_data: {}".format(test_data.shape))
    return test_data

def Debug(test_data):
    t0_date_index = feature.COL_TRADE_DATE(0)
    t0_tscode_index = feature.COL_ACTURE_OFFSET(0) + feature.ACTURE_DATA_INDEX_TSCODE
    sample_data = test_data[test_data[:,t0_date_index] == 20170103.0].copy()
    print("sample_data: {}".format(sample_data.shape))
    print("\n")
    for iloop in range(0, len(sample_data)):
        print("%f" % sample_data[iloop, t0_tscode_index])

def GetDailyDataSet(start_date):
    dataset = np.load(FileNameFixDataSetDaily())
    pos = dataset[:,feature.COL_TRADE_DATE(0)] >= start_date
    dataset = dataset[pos]
    print("test_data: {}".format(dataset.shape))
    # raw_input("Enter ...")

    pos = dataset[:,feature.COL_TRADE_DATE(feature.active_label_day)] == feature.INVALID_DATE
    print("unfinished num: %u" % np.sum(pos))
    return dataset


if __name__ == "__main__":
    # 生成测试集和训练集数据
    UpdateFixDataSet(False, False)
    CheckFixDataSet()

    # 生成 daily dataset 用于预测
    UpdateFixDataSet(True, True)
    
    # test_data = GetTestData()
    # Debug(test_data)


