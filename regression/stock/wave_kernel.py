# -*- coding:UTF-8 -*-

import tensorflow as tf
from tensorflow import keras
import tushare as ts
import numpy as np
import pandas as pd
import os
import time
import sys
import tushare_data
import random
import daily_data
import wave_test_daily

reload(sys)
sys.setdefaultencoding('utf-8')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

min_wave_width_left = 2
min_wave_width_right = 2
trade_off_threshold = 0
up_100avg_condition = True
up_200avg_condition = False
wave_index = 'close'

EXTREME_NONE = 0
EXTREME_PEAK = 1
EXTREME_VALLEY = 2

STATUS_NONE = 0
STATUS_UP = 1
STATUS_DOWN = 2

train_data_end_date = 20180101
test_data_end_date = 20190501
wave_test_dataset_sample_num = 5

def FillWaveData(input_pp_data, wave_status, start_day_index):
    for day_loop in range(start_day_index, len(input_pp_data)):
        input_pp_data.loc[day_loop,'wave_extreme'] = EXTREME_NONE
        input_pp_data.loc[day_loop,'wave_status'] = wave_status

def AppendWaveData(input_pp_data):
    if len(input_pp_data) == 0:
        return
    input_pp_data['wave_extreme'] = EXTREME_NONE
    input_pp_data['wave_status'] = STATUS_NONE
    last_extreme = EXTREME_NONE
    current_status = STATUS_NONE
    extreme_count = 0
    day_index_reversed = 0
    data_len = len(input_pp_data)
    for day_loop in reversed(range(0, data_len)):
        day_index_reversed = data_len - day_loop - 1
        if (day_index_reversed < min_wave_width_left):
            continue
        if (day_loop >= min_wave_width_right):
            # 计算 middle_value 是否是波峰或波谷
            middle_value = input_pp_data.loc[day_loop, wave_index]
            is_peak = True
            is_valley = True
            for iloop in range(1, min_wave_width_left + 1):
                temp_value = input_pp_data.loc[day_loop + iloop, wave_index]
                if middle_value > temp_value:
                    is_valley = False
                elif middle_value < temp_value:
                    is_peak = False
            for iloop in range(1, min_wave_width_right + 1):
                temp_value = input_pp_data.loc[day_loop - iloop, wave_index]
                if middle_value >= temp_value:
                    is_valley = False
                elif middle_value <= temp_value:
                    is_peak = False
            # 计算 extreme_flag 当天峰谷标志
            if is_peak:
                extreme_flag = EXTREME_PEAK
            elif is_valley:
                extreme_flag = EXTREME_VALLEY
            else:
                extreme_flag = EXTREME_NONE
        else:
            extreme_flag = EXTREME_NONE

        # 计算 current_status 当天波动状态
        if extreme_flag != EXTREME_NONE:
            last_extreme = extreme_flag
            current_status = STATUS_NONE
            if extreme_count == 0:
                FillWaveData(input_pp_data, extreme_flag, day_loop + 1)
            extreme_count += 1
        else:
            if last_extreme == EXTREME_PEAK:
                current_status = STATUS_DOWN
            elif last_extreme == EXTREME_VALLEY:
                current_status = STATUS_UP
            else:
                current_status = STATUS_NONE

        # 对 input_pp_data 赋值
        input_pp_data.loc[day_loop,'wave_extreme'] = extreme_flag
        input_pp_data.loc[day_loop,'wave_status'] = current_status

TRADE_NONE = 0
TRADE_ON = 1
TRADE_OFF = 2
train_data_list = []

def COL_INCREASE():
    return tushare_data.feature_size

def COL_TS_CODE():
    return tushare_data.feature_size + 1

def COL_ON_PRETRADE_DATE():
    return tushare_data.feature_size + 2

def COL_ON_DATE():
    return tushare_data.feature_size + 3

def COL_OFF_DATE():
    return tushare_data.feature_size + 4

def COL_HOLDING_DAYS():
    return tushare_data.feature_size + 5

def GetTrainDataUnit(pp_data, pre_on_day_index, on_date_index, off_date_index, holding_days, increase):
    if (len(pp_data) - pre_on_day_index) < 10:
        return
    data_unit=[]
    # feature
    result = tushare_data.AppendFeature(pp_data, pre_on_day_index, data_unit)
    if not result:
        return
    # label
    data_unit.append(increase)
    # ts_code
    temp_str = pp_data['ts_code'][0]
    data_unit.append(float(temp_str[0:6]))

    # 预买入日期
    temp_str = pp_data['trade_date'][pre_on_day_index]
    data_unit.append(float(temp_str))

    # 买入日期
    if on_date_index < 0:
        data_unit.append(20990101.0)
    else:
        temp_str = pp_data['trade_date'][on_date_index]
        data_unit.append(float(temp_str))

    # 卖出日期
    if off_date_index < 0:
        data_unit.append(20990101.0)
    else:
        temp_str = pp_data['trade_date'][off_date_index]
        data_unit.append(float(temp_str))

    # 持有天数
    data_unit.append(float(holding_days))

    # 添加到数据集列表
    train_data_list.append(data_unit)

model=keras.models.load_model("./model/model.h5")
mean=np.load('./model/mean.npy')
std=np.load('./model/std.npy')

def Predict(pp_data, day_index):
    if (len(pp_data) - day_index) < 10:
        return 0.0
    data_unit=[]
    tushare_data.AppendFeature(pp_data, day_index, data_unit)
    test_data = []
    test_data.append(data_unit)
    predict_features = np.array(test_data)
    predict_features = (predict_features - mean) / std
    prediction = model.predict(predict_features)
    return prediction[0]


def FileNameDataSet():
    file_name = './data/dataset/dataset_%u_%s_%s_%s_%s_%s_%u_%u_%u_%u_%u_%u.npy' % ( \
        tushare_data.feature_type, \
        tushare_data.stocks_list_end_date, \
        tushare_data.pp_data_start_date, \
        tushare_data.train_test_date, \
        tushare_data.industry_filter, \
        tushare_data.code_filter, \
        min_wave_width_left, \
        min_wave_width_right, \
        trade_off_threshold, \
        int(up_100avg_condition), \
        int(up_200avg_condition), \
        wave_test_dataset_sample_num)
    return file_name

def FileNameDataSetOriginal():
    file_name = './data/dataset/dataset_original_%u_%s_%s_%s_%s_%s_%u_%u_%u_%u_%u_%u.npy' % ( \
        tushare_data.feature_type, \
        tushare_data.stocks_list_end_date, \
        tushare_data.pp_data_start_date, \
        tushare_data.train_test_date, \
        tushare_data.industry_filter, \
        tushare_data.code_filter, \
        min_wave_width_left, \
        min_wave_width_right, \
        trade_off_threshold, \
        int(up_100avg_condition), \
        int(up_200avg_condition), \
        wave_test_dataset_sample_num)
    return file_name

def FileNameDailyDataSet():
    file_name = './data/dataset/daily_dataset_%u_%s_%s_%s_%s_%s_%s_%u_%u_%u_%u_%u_%u_%u.npy' % ( \
        tushare_data.feature_type, \
        tushare_data.stocks_list_end_date, \
        tushare_data.pp_data_start_date, \
        tushare_data.industry_filter, \
        tushare_data.code_filter, \
        daily_data.start_date, \
        daily_data.end_date, \
        min_wave_width_left, \
        min_wave_width_right, \
        trade_off_threshold, \
        int(up_100avg_condition), \
        int(up_200avg_condition), \
        int(wave_test_daily.pridect_mode), \
        wave_test_dataset_sample_num)
    return file_name

def FileNameDailyDataSetOriginal():
    file_name = './data/dataset/daily_dataset_original_%u_%s_%s_%s_%s_%s_%s_%u_%u_%u_%u_%u_%u_%u.npy' % ( \
        tushare_data.feature_type, \
        tushare_data.stocks_list_end_date, \
        tushare_data.pp_data_start_date, \
        tushare_data.industry_filter, \
        tushare_data.code_filter, \
        daily_data.start_date, \
        daily_data.end_date, \
        min_wave_width_left, \
        min_wave_width_right, \
        trade_off_threshold, \
        int(up_100avg_condition), \
        int(up_200avg_condition), \
        int(wave_test_daily.pridect_mode), \
        wave_test_dataset_sample_num)
    return file_name

def AppendPreTradeStockNums(data_set):
    print('AppendPreTradeStockNums.Start')
    data_set = data_set[np.where(data_set[:,COL_TS_CODE()] != 000029.0)].copy()
    # 获取 data_set 的最大和最小 pre_on 时间，生成date_list列表
    on_pretrade_dates = data_set[:, COL_ON_PRETRADE_DATE()]
    start_date = '%.0f' % np.min(on_pretrade_dates)
    end_date = '%.0f' % np.max(on_pretrade_dates)
    date_list = tushare_data.TradeDateListRange(start_date, end_date)
    
    # 生成 np_date_holding_num，二列表格
    holding_nums_list = []
    avg_sample_num = 5
    for iloop in range(0, len(date_list) - avg_sample_num):
        temp_date = int(date_list[iloop])
        temp_date_range_b = int(date_list[iloop + avg_sample_num])
        pos = (on_pretrade_dates > temp_date_range_b) & (on_pretrade_dates <= temp_date)
        holding_nums = np.sum(pos)
        temp_unit = []
        temp_unit.append(temp_date)
        temp_unit.append(holding_nums)
        holding_nums_list.append(temp_unit)
    np_date_holding_num = np.array(holding_nums_list)

    # 生成每条记录的 holding_nums 列表 - dataset_holding_nums_list
    data_set = data_set[np.where(data_set[:,COL_ON_PRETRADE_DATE()] > int(date_list[len(date_list) - avg_sample_num]))].copy()
    on_pretrade_dates = data_set[:, COL_ON_PRETRADE_DATE()]
    dataset_holding_nums_list = []
    for iloop in range(0, len(data_set)):
        temp_date = int(on_pretrade_dates[iloop])
        temp_record = np_date_holding_num[np_date_holding_num[:, 0] == temp_date]
        if len(temp_record) != 1:
            print('Error, AppendPreTradeStockNums, len(temp_record) != 1')
            return
        temp_num = float(temp_record[0][1])
        dataset_holding_nums_list.append(temp_num)
    dst_col = np.array(dataset_holding_nums_list)
    dst_col = dst_col.reshape((len(data_set), 1))
    merge_dataset = np.append(dst_col, data_set, axis = 1)
    print('AppendPreTradeStockNums.Finish')
    return merge_dataset

def AppendHoldStockNums(data_set):
    return AppendPreTradeStockNums(data_set)
    print('AppendHoldStockNums.Start')
    data_set = data_set[np.where(data_set[:,COL_TS_CODE()] != 000029.0)].copy()
    # 获取 data_set 的最大和最小 pre_on 时间，生成date_list列表
    on_pretrade_dates = data_set[:, COL_ON_PRETRADE_DATE()]
    start_date = '%.0f' % np.min(on_pretrade_dates)
    end_date = '%.0f' % np.max(on_pretrade_dates)
    date_list = tushare_data.TradeDateListRange(start_date, end_date)
    
    # 生成 np_date_holding_num，二列表格
    on_dates = data_set[:, COL_ON_DATE()]
    off_dates = data_set[:, COL_OFF_DATE()]
    holding_nums_list = []
    for iloop in range(0, len(date_list)):
        temp_date = int(date_list[iloop])
        pos = (on_dates <= temp_date) & (off_dates > temp_date)
        holding_nums = np.sum(pos)
        temp_unit = []
        temp_unit.append(temp_date)
        temp_unit.append(holding_nums)
        holding_nums_list.append(temp_unit)
    np_date_holding_num = np.array(holding_nums_list)

    # 生成每条记录的 holding_nums 列表 - dataset_holding_nums_list
    dataset_holding_nums_list = []
    for iloop in range(0, len(data_set)):
        temp_date = int(on_pretrade_dates[iloop])
        temp_record = np_date_holding_num[np_date_holding_num[:, 0] == temp_date]
        if len(temp_record) != 1:
            print('Error, AppendHoldStockNums, len(temp_record) != 1')
            return
        temp_num = float(temp_record[0][1])
        dataset_holding_nums_list.append(temp_num)
    dst_col = np.array(dataset_holding_nums_list)
    dst_col = dst_col.reshape((len(data_set), 1))
    merge_dataset = np.append(dst_col, data_set, axis = 1)
    print('AppendHoldStockNums.Finish')
    return merge_dataset

def SaveDataSet():
    train_data = np.array(train_data_list)
    np.save(FileNameDataSetOriginal(), train_data)
    train_data = AppendHoldStockNums(train_data)
    np.save(FileNameDataSet(), train_data)

def GetTrainData():
    train_data = np.load(FileNameDataSet())
    tushare_data.feature_size += 1
    print("data_set: {}".format(train_data.shape))

    pos = (train_data[:,COL_ON_PRETRADE_DATE()] < train_data_end_date) & (train_data[:,COL_OFF_DATE()] != 20990101.0)
    train_data = train_data[pos].copy()
    print("train_data: {}".format(train_data.shape))

    print("reorder...")
    order=np.argsort(np.random.random(len(train_data)))
    train_data=train_data[order]
    train_data=train_data[:2000000]

    label_index = tushare_data.feature_size
    print("get feature ...")
    train_features = train_data[:, 0:tushare_data.feature_size].copy()
    # raw_input("Enter ...")

    print("get label...")
    train_labels = train_data[:, label_index:label_index+1].copy()
    # raw_input("Enter ...")
    print("train_features: {}".format(train_features.shape))
    print("train_labels: {}".format(train_labels.shape))

    return train_features, train_labels

def GetTrainDataOriginal():
    train_data = np.load(FileNameDataSetOriginal())
    print("data_set: {}".format(train_data.shape))

    pos = (train_data[:,COL_ON_PRETRADE_DATE()] < train_data_end_date) & (train_data[:,COL_OFF_DATE()] != 20990101.0)
    train_data = train_data[pos].copy()
    print("train_data: {}".format(train_data.shape))

    print("reorder...")
    order=np.argsort(np.random.random(len(train_data)))
    train_data=train_data[order]
    train_data=train_data[:2000000]

    label_index = tushare_data.feature_size
    print("get feature ...")
    train_features = train_data[:, 0:tushare_data.feature_size].copy()
    # raw_input("Enter ...")

    print("get label...")
    train_labels = train_data[:, label_index:label_index+1].copy()
    # raw_input("Enter ...")
    print("train_features: {}".format(train_features.shape))
    print("train_labels: {}".format(train_labels.shape))

    return train_features, train_labels

def GetTestData():
    data_set = np.load(FileNameDataSet())

    # debug
    # tushare_data.feature_size += 1
    # captions = []
    # for iloop in range(0, tushare_data.feature_size):
    #     captions.append('f_%u' % iloop)
    # captions.append('label')
    # captions.append('ts_code')
    # captions.append('pre_on_date')
    # captions.append('on_date')
    # captions.append('off_date')
    # captions.append('holding_days')
    # data_df = pd.DataFrame(data_set, columns=captions)
    # data_df = data_df.sort_values(by=['pre_on_date', 'ts_code'], ascending=(True, True))

    # pos = (data_df['on_date'] <= 20190215.0) & (data_df['off_date'] > 20190215.0) & (data_df['pre_on_date'] < 20190101.0)
    # print('pos: %u' % np.sum(pos))
    # debug_df = data_df[pos]
    # print(debug_df)
    # return debug_df

    tushare_data.feature_size += 1
    print("data_set: {}".format(data_set.shape))
    pos = (data_set[:,COL_ON_PRETRADE_DATE()] >= train_data_end_date) & (data_set[:,COL_ON_PRETRADE_DATE()] < test_data_end_date)
    data_set = data_set[pos]
    print("test_data: {}".format(data_set.shape))

    captions = []
    for iloop in range(0, tushare_data.feature_size):
        captions.append('f_%u' % iloop)
    captions.append('label')
    captions.append('ts_code')
    captions.append('pre_on_date')
    captions.append('on_date')
    captions.append('off_date')
    captions.append('holding_days')
    data_df = pd.DataFrame(data_set, columns=captions)
    data_df = data_df.sort_values(by=['pre_on_date', 'ts_code'], ascending=(True, True))

    # debug_df = data_df[data_df['pre_on_date'] == 20190215.0]
    # print(debug_df)

    # pos = (data_df['on_date'] <= 20190215.0) & (data_df['off_date'] > 20190215.0)
    
    # print('pos: %u' % np.sum(pos))
    # debug_df = data_df[pos]
    # print(debug_df)
    # return debug_df.values

    return data_df.values

    data_set = data_set[np.argsort(data_set[:,tushare_data.feature_size + 2])]
    return data_set

def GetTestDataOriginal():
    data_set = np.load(FileNameDataSet())
    data_set = data_set[:,1:]
    print("data_set: {}".format(data_set.shape))
    pos = (data_set[:,COL_ON_PRETRADE_DATE()] >= train_data_end_date) & (data_set[:,COL_ON_PRETRADE_DATE()] < test_data_end_date)
    data_set = data_set[pos]
    print("test_data: {}".format(data_set.shape))
    captions = []
    for iloop in range(0, tushare_data.feature_size):
        captions.append('f_%u' % iloop)
    captions.append('label')
    captions.append('ts_code')
    captions.append('pre_on_date')
    captions.append('on_date')
    captions.append('off_date')
    captions.append('holding_days')
    data_df = pd.DataFrame(data_set, columns=captions)
    data_df = data_df.sort_values(by=['pre_on_date', 'ts_code'], ascending=(True, True))
    return data_df.values

def SaveDailyDataSet():
    train_data = np.array(train_data_list)
    np.save(FileNameDailyDataSetOriginal(), train_data)
    train_data = AppendHoldStockNums(train_data)
    np.save(FileNameDailyDataSet(), train_data)

def GetDailyDataSet():
    data_set = np.load(FileNameDailyDataSet())
    tushare_data.feature_size += 1
    print("data_set: {}".format(data_set.shape))
    data_set = data_set[np.where(data_set[:,tushare_data.feature_size + 2] >= train_data_end_date)]
    print("data_set from train_data_end_date: {}".format(data_set.shape))

    # temp_data = data_set[data_set[:, tushare_data.feature_size + 1] == 000012.0]
    # print[temp_data]
    # return temp_data

    captions = []
    for iloop in range(0, tushare_data.feature_size):
        captions.append('f_%u' % iloop)
    captions.append('label')
    captions.append('ts_code')
    captions.append('pre_on_date')
    captions.append('on_date')
    captions.append('off_date')
    captions.append('holding_days')
    data_df = pd.DataFrame(data_set, columns=captions)
    data_df = data_df.sort_values(by=['pre_on_date', 'ts_code'], ascending=(True, True))

    # debug_df = data_df[data_df['pre_on_date'] == 20190214.0]
    # print(debug_df)
    # return debug_df.values

    return data_df.values

    # data_set = data_set[np.argsort(data_set[:,tushare_data.feature_size + 2])]
    # return data_set

def PrintRecord(trade_count, \
                ts_code, \
                on_date, \
                off_date, \
                holding_days, \
                on_price, \
                off_price):
    if holding_days >= 0:
        str_holding_days = '%u' % holding_days
    else:
        str_holding_days = '--'

    if on_price > 0:
        str_on_price = '%.2f' % on_price
    else:
        str_on_price = '--'

    if off_price > 0:
        str_off_price = '%.2f' % off_price
    else:
        str_off_price = '--'

    if on_price > 0 and off_price > 0:
        increase = ((off_price / on_price) - 1.0) * 100.0
        str_increase = '%.2f' % increase
    else:
        str_increase = '--'
    print("%-6u%-10s%-12s%-12s%-10s%-10s%-10s%-10s" %( \
        trade_count, \
        ts_code, \
        on_date, \
        off_date, \
        str_holding_days, \
        str_on_price, \
        str_off_price, \
        str_increase))

def TradeTestFinishedHandel(trade_count, \
                            input_pp_data, \
                            pre_on_day_index, \
                            on_day_index, \
                            pre_off_day_index, \
                            off_day_index, \
                            print_record, \
                            save_data_set):
    ts_code = input_pp_data.loc[0, 'ts_code']
    on_price = input_pp_data.loc[on_day_index, 'open']
    on_date = input_pp_data.loc[on_day_index,'trade_date']
    off_price = input_pp_data.loc[off_day_index, 'open']
    off_date = input_pp_data.loc[off_day_index,'trade_date']
    holding_days = on_day_index - off_day_index
    increase = ((off_price / on_price) - 1.0) * 100.0
    if print_record:
        PrintRecord(trade_count, \
            ts_code, \
            on_date, \
            off_date, \
            holding_days, \
            on_price, \
            off_price)
    if save_data_set:
        if holding_days > wave_test_dataset_sample_num:
            temp_sample_num = wave_test_dataset_sample_num
        else:
            temp_sample_num = holding_days
        for iloop in range(0, temp_sample_num):
            temp_on_price = input_pp_data.loc[on_day_index - iloop, 'open']
            temp_increase = ((off_price / temp_on_price) - 1.0) * 100.0
            GetTrainDataUnit(input_pp_data, \
                pre_on_day_index - iloop, \
                on_day_index - iloop, \
                off_day_index, \
                holding_days - iloop, \
                temp_increase)
    return increase, holding_days
    
def TradeTestUnfinishedPreOnHandel(trade_count, \
                            input_pp_data, \
                            pre_on_day_index, \
                            trade_offset, \
                            print_record, \
                            save_data_set):
    ts_code = input_pp_data.loc[0, 'ts_code']
    on_date = '%s+%u' % (input_pp_data.loc[pre_on_day_index,'trade_date'], trade_offset)
    if print_record:
        PrintRecord(trade_count, \
            ts_code, \
            on_date, \
            '--', \
            -1, \
            -1, \
            -1)
    if save_data_set:
        GetTrainDataUnit(input_pp_data, \
            pre_on_day_index, \
            -1, \
            -1, \
            0, \
            0.0)

def TradeTestUnfinishedPreOffHandel(trade_count, \
                            input_pp_data, \
                            pre_on_day_index, \
                            on_day_index, \
                            pre_off_day_index, \
                            trade_offset, \
                            print_record, \
                            save_data_set):
    ts_code = input_pp_data.loc[0, 'ts_code']
    on_price = input_pp_data.loc[on_day_index, 'open']
    on_date = input_pp_data.loc[on_day_index,'trade_date']
    off_date = '%s+%u' % (input_pp_data.loc[pre_off_day_index,'trade_date'], trade_offset)
    holding_days = on_day_index - pre_off_day_index + trade_offset
    if print_record:
        PrintRecord(trade_count, \
            ts_code, \
            on_date, \
            off_date, \
            holding_days, \
            on_price, \
            -1)
    if save_data_set:
        GetTrainDataUnit(input_pp_data, \
            pre_on_day_index, \
            on_day_index, \
            -1, \
            holding_days, \
            0.0)

def TradeTestUnfinishedHandel(trade_count, \
                            input_pp_data, \
                            pre_on_day_index, \
                            on_day_index, \
                            print_record, \
                            save_data_set):
    ts_code = input_pp_data.loc[0, 'ts_code']
    on_price = input_pp_data.loc[on_day_index, 'open']
    on_date = input_pp_data.loc[on_day_index,'trade_date']
    holding_days = on_day_index - 0 + 1
    if print_record:
        PrintRecord(trade_count, \
            ts_code, \
            on_date, \
            '--', \
            holding_days, \
            on_price, \
            -1)
    if save_data_set:
        if holding_days > wave_test_dataset_sample_num:
            temp_sample_num = wave_test_dataset_sample_num
        else:
            temp_sample_num = holding_days
        for iloop in range(0, temp_sample_num):
            GetTrainDataUnit(input_pp_data, \
                pre_on_day_index - iloop, \
                on_day_index -iloop, \
                -1, \
                holding_days -iloop, \
                0.0)


def TradeTest(input_pp_data, \
              cut_loss_ratio, \
              print_finished_record, \
              print_unfinished_record, \
              print_trade_flag, \
              print_summary, \
              save_finished_dataset = False, \
              save_unfinished_dataset = False):
    if len(input_pp_data) == 0:
        return 0.0, 0
    ts_code = input_pp_data.loc[0, 'ts_code']
    input_pp_data['wave_trade'] = TRADE_NONE
    data_len = len(input_pp_data)
    current_trade_status = TRADE_OFF
    last_peak = -1.0
    last_valley = -1.0
    day_index = data_len - 1
    pre_on_day_index = 0
    pre_off_day_index = 0
    on_day_index = 0
    off_day_index = 0
    on_price = 0.0
    off_price = 0.0
    sum_increase = 0.0
    sum_holding_days = 0
    trade_count = 0
    trade_count_profitable = 0
    test_days = data_len
    trade_off_count = 0
    while day_index >= 0:
        current_date = input_pp_data.loc[day_index, 'trade_date']
        close = input_pp_data.loc[day_index, wave_index]
        close_10_avg = input_pp_data.loc[day_index, 'close_10_avg']
        close_100_avg = input_pp_data.loc[day_index, 'close_100_avg']
        close_200_avg = input_pp_data.loc[day_index, 'close_200_avg']
        wave_extreme = input_pp_data.loc[day_index, 'wave_extreme']
        wave_status = input_pp_data.loc[day_index, 'wave_status']
        trade_flag = TRADE_NONE
        if last_peak > 0.0 and last_valley > 0.0:
            if current_trade_status == TRADE_OFF:
                # ON 事件产生
                if (close > last_peak) and \
                   ((not up_100avg_condition) or (close_10_avg > close_100_avg)) and \
                   ((not up_200avg_condition) or (close_10_avg > close_200_avg)) and \
                   (trade_off_count > trade_off_threshold):
                    
                    trade_day_offset = 1
                    trade_flag = TRADE_ON
                    on_reason = 1
                        
                # elif wave_extreme == EXTREME_VALLEY and close > last_valley:
                #     # print('%s,wave_extreme == EXTREME_VALLEY and close > last_valley:%f>%f' % (current_date, close, last_valley))
                #     trade_day_offset = min_wave_width_right + 1
                #     trade_flag = TRADE_ON
                #     on_reason = 2

                # ON 事件处理
                if trade_flag == TRADE_ON:
                    pre_on_day_index = day_index
                    if day_index >= trade_day_offset:
                        day_index -= trade_day_offset
                        on_day_index = day_index
                        on_price = input_pp_data.loc[day_index, 'open']
                        current_trade_status = TRADE_ON
                        on_date = input_pp_data.loc[day_index,'trade_date']
                    else:
                        # 预买入信号
                        TradeTestUnfinishedPreOnHandel(trade_count, \
                                                    input_pp_data, \
                                                    pre_on_day_index, \
                                                    trade_day_offset, \
                                                    print_trade_flag, \
                                                    save_unfinished_dataset)
            elif current_trade_status == TRADE_ON:
                # OFF 事件产生
                if wave_extreme == EXTREME_PEAK and close <= last_peak: 
                    trade_day_offset = min_wave_width_right + 1
                    trade_flag = TRADE_OFF
                    off_reason = 1
                elif wave_extreme == EXTREME_VALLEY and close <= last_valley: 
                    trade_day_offset = min_wave_width_right + 1
                    trade_flag = TRADE_OFF
                    off_reason = 2
                elif wave_status == STATUS_DOWN and close < (on_price * (1.0 - cut_loss_ratio)):
                    trade_day_offset = 1
                    trade_flag = TRADE_OFF
                    off_reason = 3
                elif close < last_valley:
                    trade_day_offset = 1
                    trade_flag = TRADE_OFF
                    off_reason = 4
                # OFF 事件处理
                if trade_flag == TRADE_OFF:
                    pre_off_day_index = day_index
                    if day_index >= trade_day_offset:
                        day_index -= trade_day_offset
                        off_day_index = day_index
                        current_trade_status = TRADE_OFF
                        temp_increase,temp_holding_days = TradeTestFinishedHandel(trade_count, \
                                                                                    input_pp_data, \
                                                                                    pre_on_day_index, \
                                                                                    on_day_index, \
                                                                                    pre_off_day_index, \
                                                                                    off_day_index, \
                                                                                    print_finished_record, \
                                                                                    save_finished_dataset)
                        sum_increase += temp_increase
                        sum_holding_days += temp_holding_days
                        trade_count += 1
                        if temp_increase > 0:
                            trade_count_profitable += 1
                    else:
                        # 预卖出信号
                        if print_trade_flag:
                            TradeTestUnfinishedPreOffHandel(trade_count, \
                                                            input_pp_data, \
                                                            pre_on_day_index, \
                                                            on_day_index, \
                                                            pre_off_day_index, \
                                                            trade_day_offset, \
                                                            print_trade_flag, \
                                                            save_unfinished_dataset)
        # 更新 last_peak 和 last_valley
        if wave_extreme == EXTREME_PEAK:
            last_peak = close
        elif wave_extreme == EXTREME_VALLEY:
            last_valley = close

        day_index -= 1
        if current_trade_status == TRADE_ON:
            trade_off_count = 0
        else:
            trade_off_count += 1
    if current_trade_status == TRADE_ON:
        # 还没有结束的交易
        TradeTestUnfinishedHandel(trade_count, \
                            input_pp_data, \
                            pre_on_day_index, \
                            on_day_index, \
                            print_unfinished_record, \
                            save_unfinished_dataset)
    if print_summary:
        print("test_days: %u, holding_days_sum: %u, increase: %.2f" % (test_days, sum_holding_days, sum_increase))
    return sum_increase, sum_holding_days, trade_count, trade_count_profitable
    

if __name__ == "__main__":
    # df = GetDownloadMergeData()
    # stock_df = GetDownloadData(df, '000529.SZ')

    # df = daily_data.GetPreprocessedMergeData()
    # stock_df = daily_data.GetPreprocessedData(df, '000529.SZ')
    # AppendWaveData(stock_df)
    # stock_df.to_csv('./temp.csv')
    # TradeTest(stock_df, 0.05, True, True, True, True, False, False)

    # DailyDataSetAppendHoldStockNums()
    # data_set = GetDailyDataSet()
    # print(data_set)

    if os.path.exists(FileNameDailyDataSetOriginal()):
        if not os.path.exists(FileNameDailyDataSet()):
            data_set = np.load(FileNameDailyDataSetOriginal())
            print("data_set: {}".format(data_set.shape))
            data_set = AppendHoldStockNums(data_set)
            print("data_set: {}".format(data_set.shape))
            np.save(FileNameDailyDataSet(), data_set)

    if os.path.exists(FileNameDataSetOriginal()):
        if not os.path.exists(FileNameDataSet()):
            data_set = np.load(FileNameDataSetOriginal())
            print("data_set: {}".format(data_set.shape))
            data_set = AppendHoldStockNums(data_set)
            print("data_set: {}".format(data_set.shape))
            np.save(FileNameDataSet(), data_set)