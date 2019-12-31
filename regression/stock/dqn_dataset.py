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

def SettingNameCode():
    temp_name = '%s_%u' % ( \
        feature.SettingName(), \
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

def FileNameDataSetSplit(ts_code):
    file_name = './data/dataset/dqn_split/%s_%s_%s.npy' % (SettingNameCode(), tushare_data.train_test_date, ts_code)
    return file_name

def CreateDataSetSplit():
    start_date = dataset_start_date
    end_date = tushare_data.train_test_date
    date_list = tushare_data.TradeDateListRange(start_date, end_date).tolist()
    code_list = tushare_data.StockCodes(dataset_stock_sample_step)
    date_index_map = ListToIndexMap(date_list, True)

    data_unit_date_index = ACTURE_DATA_INDEX_DATE()
    valid_data_unit_num = 0
    for code_index in range(0, len(code_list)):
        stock_code = code_list[code_index]
        dataset_split_file_name = FileNameDataSetSplit(stock_code)
        if not os.path.exists(dataset_split_file_name):
            stock_pp_file_name = tushare_data.FileNameStockPreprocessedData(stock_code)
            if os.path.exists(stock_pp_file_name):
                pp_data = pd.read_csv(stock_pp_file_name)
            else:
                pp_data = []
            if len(pp_data) == 0:
                continue
            
            dataset = np.zeros((len(date_list), 1, feature.feature_size + feature.acture_unit_size))
            for day_loop in range(0, len(pp_data)):
                data_unit = feature.GetDataUnit1Day(pp_data, day_loop)
                if len(data_unit) == 0:
                    continue
                temp_date = int(data_unit[data_unit_date_index])
                if temp_date < start_date or temp_date > end_date:
                    continue
                dateset_index1 = date_index_map[pp_data['trade_date'][day_loop]]
                dataset[dateset_index1][0] = data_unit

            split_data_date = dataset[:,:,ACTURE_DATA_INDEX_DATE()]
            np.save(dataset_split_file_name, dataset)
        print("%-4d : %s 100%%" % (code_index, stock_code))

def CreateDataSetMerge(dataset_file_name):
    start_date = dataset_start_date
    end_date = tushare_data.train_test_date
    date_list = tushare_data.TradeDateListRange(start_date, end_date).tolist()
    code_list = tushare_data.StockCodes(dataset_stock_sample_step)
    dataset = np.zeros((len(date_list), len(code_list), feature.feature_size + feature.acture_unit_size))
    code_index_map = ListToIndexMap(code_list)
    for code_index in range(0, len(code_list)):
        stock_code = code_list[code_index]
        dataset_split_file_name = FileNameDataSetSplit(stock_code)
        if not os.path.exists(dataset_split_file_name):
            continue
        split_data = np.load(dataset_split_file_name)
        dateset_index2 = code_index_map[stock_code]
        for iloop in range(len(date_list)):
            dataset[iloop][dateset_index2] = split_data[iloop][0]
    print("dataset: {}".format(dataset.shape))
    print("file_name: %s" % dataset_file_name)
    np.save(dataset_file_name, dataset)


def CreateDataSet():
    dataset_file_name = FileNameDataSet(False)
    if os.path.exists(dataset_file_name):
        print('dataset already exist: %s' % dataset_file_name)
        return
    CreateDataSetSplit()
    CreateDataSetMerge(dataset_file_name)
    


def GetDataSet():
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
    # GetDataSet()

    # dataset_ = np.load('./data/dataset/dqn_0_30_1_10_0_0___20000101_10_20020101_20190414_.npy')
    # dataset = np.load('./data/dataset/dqn_0_30_1_10_0_0___20000101_10_20020101_20190414.npy')

    dataset_ = np.load('./data/dataset/dqn_fix_0_30_1_10_0_0___20000101_10_20020101_20190414_20170101_10_0.800000_.npy')
    dataset = np.load('./data/dataset/dqn_fix_0_30_1_10_0_0___20000101_10_20020101_20190414_20170101_10_0.800000.npy')

    d_ = dataset_.flatten()
    d = dataset.flatten()
    print('d_:{}, {}'.format(d_.shape, d_.dtype))
    print('d:{}, {}'.format(d.shape, d.dtype))
    for iloop in range(len(d)):
        if (float(d_[iloop]) - float(d[iloop])) > 0.000001:
            print('%-16u%-16.10f%-16.10f' % (iloop, d_[iloop], d[iloop]))
    



