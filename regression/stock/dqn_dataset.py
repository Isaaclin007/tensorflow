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

dataset_start_date = 20020101
dataset_train_test_split_date = 20170101
dataset_stock_sample_step = 10

def SettingName():
    temp_name = '%s_%s_%s_%s_%u_%u' % ( \
        feature.SettingName(), \
        tushare_data.industry_filter, \
        tushare_data.code_filter, \
        tushare_data.stocks_list_end_date, \
        dataset_stock_sample_step, \
        dataset_start_date)
    return temp_name

def TrainSettingName():
    temp_name = '%s_%u' % (SettingName(), dataset_train_test_split_date)
    return temp_name

def FileNameDataSet(is_daily_data=False):
    if is_daily_data:
        file_name = './data/dataset/dqn_daily_%s_%s.npy' % (SettingName(), pp_daily_update.update_date)
    else:
        file_name = './data/dataset/dqn_%s_%s.npy' % (SettingName(), tushare_data.train_test_date)
    return file_name

def ListToIndexMap(input_list, to_int_value=False):
    temp_dict = {}
    for iloop in range(0, len(input_list)):
        if to_int_value:
            temp_dict[int(input_list[iloop])] = iloop
        else:
            temp_dict[input_list[iloop]] = iloop
    return temp_dict

def ACTURE_DATA_INDEX_DATE():
    return feature.feature_size + feature.ACTURE_DATA_INDEX_DATE

def ACTURE_DATA_INDEX_OPEN():
    return feature.feature_size + feature.ACTURE_DATA_INDEX_OPEN

def ACTURE_DATA_INDEX_TSCODE():
    return feature.feature_size + feature.ACTURE_DATA_INDEX_TSCODE

def CreateDataSet():
    dataset_file_name = FileNameDataSet(False)
    if os.path.exists(dataset_file_name):
        print('dataset already exist: %s' % dataset_file_name)
        return
    start_date = dataset_start_date
    end_date = tushare_data.train_test_date
    date_list = tushare_data.TradeDateListRange(start_date, end_date).tolist()
    code_list = tushare_data.StockCodes(dataset_stock_sample_step)
    date_index_map = ListToIndexMap(date_list, True)
    code_index_map = ListToIndexMap(code_list)

    dataset = np.zeros((len(date_list), len(code_list), feature.feature_size + feature.acture_unit_size), dtype=float)
    data_unit_date_index = ACTURE_DATA_INDEX_DATE()
    valid_data_unit_num = 0
    for code_index in range(0, len(code_list)):
        stock_code = code_list[code_index]
        stock_pp_file_name = tushare_data.FileNameStockPreprocessedData(stock_code)
        if os.path.exists(stock_pp_file_name):
            pp_data = pd.read_csv(stock_pp_file_name)
        else:
            pp_data = []
        if len(pp_data) == 0:
            continue
        for day_loop in range(0, len(pp_data)):
            data_unit = feature.GetDataUnit1Day(pp_data, day_loop)
            if len(data_unit) == 0:
                continue
            temp_date = int(data_unit[data_unit_date_index])
            if temp_date < start_date or temp_date > end_date:
                continue
            dateset_index1 = date_index_map[pp_data['trade_date'][day_loop]]
            dateset_index2 = code_index_map[stock_code]
            dataset[dateset_index1][dateset_index2] = data_unit
            valid_data_unit_num += 1
        print("%-4d : %s 100%%" % (code_index, stock_code))
    print("dataset: {}".format(dataset.shape))
    print("valid_data_unit_num: %u" % valid_data_unit_num)
    print("file_name: %s" % dataset_file_name)
    np.save(dataset_file_name, dataset)

def GetDataSet():
    CreateDataSet()
    dataset_file_name = FileNameDataSet(False)
    dataset = np.load(dataset_file_name)
    print("dataset: {}".format(dataset.shape))
    start_date = dataset_start_date
    end_date = tushare_data.train_test_date
    date_list = tushare_data.TradeDateListRange(start_date, end_date).tolist()
    dataset_train_test_split_index = 0
    for iloop in range(0, len(date_list)):
        # date_list 和 dataset 中的 date 都是从达到小排序
        if int(date_list[iloop]) < dataset_train_test_split_date:
            dataset_train_test_split_index = iloop
            break

    train_dataset = dataset[dataset_train_test_split_index:]
    test_dataset = dataset[:dataset_train_test_split_index]
    print("train: {}".format(train_dataset.shape))
    print("test: {}".format(test_dataset.shape))
    return train_dataset, test_dataset


if __name__ == "__main__":
    CreateDataSet()
    GetDataSet()


