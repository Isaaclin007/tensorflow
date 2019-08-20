# -*- coding:UTF-8 -*-


import tushare as ts
import numpy as np
import pandas as pd
import os
import time
import datetime
import sys
import math

# 常量定义
FEATURE_G0_D5_AVG = 0
LABEL_PRE_CLOSE_2_TD_CLOSE = 0
LABEL_T1_OPEN_2_TD_CLOSE = 1
LABEL_CONSECUTIVE_RISE_SCORE = 2
LABEL_T1_OPEN_2_TD_OPEN = 3
INVALID_DATE = 99990101.0

# 设置参数
feature_days = 30
label_days = 10
active_label_day = 1  # 0 ~ (label_days-1)
feature_type = FEATURE_G0_D5_AVG
label_type = LABEL_T1_OPEN_2_TD_CLOSE

# 计算参数
feature_unit_size = 0
if feature_type == FEATURE_G0_D5_AVG:
    feature_unit_size = 5
feature_size = feature_unit_size * feature_days
acture_unit_size = 7

ACTURE_DATA_INDEX_OPEN_INCREASE = 0
ACTURE_DATA_INDEX_LOW_INCREASE = 1
ACTURE_DATA_INDEX_OPEN = 2
ACTURE_DATA_INDEX_LOW = 3
ACTURE_DATA_INDEX_CLOSE = 4
ACTURE_DATA_INDEX_TSCODE = 5
ACTURE_DATA_INDEX_DATE = 6

def ACTIVE_LABEL_DAY():
    return active_label_day

def COL_ACTIVE_LABEL():
    return feature_size + active_label_day

def COL_FEATURE_OFFSET():
    return 0

def FEATURE_SIZE():
    return feature_size

def ACTURE_UNIT_SIZE():
    return acture_unit_size

def COL_ACTURE_OFFSET(actrue_day_index):
    return (feature_size + label_days + (actrue_day_index * acture_unit_size))

def COL_TRADE_DATE(actrue_day_index):
    return (COL_ACTURE_OFFSET(actrue_day_index) + ACTURE_DATA_INDEX_DATE)

# 从 day_index 开始的前 feature_days 天的数据，包含 day_index
def AppendFeature(pp_data, day_index, data_unit):
    if feature_type == FEATURE_G0_D5_AVG:
        base_close = pp_data['close_100_avg'][day_index]
        base_vol = pp_data['vol_100_avg'][day_index]
        for iloop in reversed(range(0, feature_days)):
            temp_index = day_index + iloop
            if (pp_data['suspend'][temp_index] != 0) or (pp_data['adj_flag'][temp_index] != 0):
                return False
            data_unit.append(pp_data['open'][temp_index] / base_close)
            data_unit.append(pp_data['close'][temp_index] / base_close)
            data_unit.append(pp_data['high'][temp_index] / base_close)
            data_unit.append(pp_data['low'][temp_index] / base_close)
            data_unit.append(pp_data['vol'][temp_index] / base_vol)
    return True

# 从 day_index-1 开始的后 label_days 天的数据，不包含 day_index
def AppendLabel(pp_data, day_index, data_unit):
    start_day_index = day_index - 1
    for iloop in range(0, label_days):
        temp_index = start_day_index - iloop
        if temp_index >= 0:
            if (pp_data['suspend'][temp_index] != 0) or (pp_data['adj_flag'][temp_index] != 0):
                return False
    if label_type == LABEL_PRE_CLOSE_2_TD_CLOSE:
        start_price = pp_data['close'][day_index]
        for iloop in range(0, label_days):
            temp_index = start_day_index - iloop
            if temp_index < 0:
                data_unit.append(0.0)
            else:
                temp_increase_per = ((pp_data['close'][temp_index] / start_price) - 1.0) * 100.0
                data_unit.append(temp_increase_per)
    elif label_type == LABEL_T1_OPEN_2_TD_CLOSE:
        start_price = pp_data['open'][start_day_index]
        for iloop in range(0, label_days):
            temp_index = start_day_index - iloop
            if temp_index < 0:
                data_unit.append(0.0)
            else:
                temp_increase_per = ((pp_data['close'][temp_index] / start_price) - 1.0) * 100.0
                data_unit.append(temp_increase_per)
    elif label_type == LABEL_T1_OPEN_2_TD_OPEN:
        start_price = pp_data['open'][start_day_index]
        for iloop in range(0, label_days):
            temp_index = start_day_index - iloop
            if temp_index < 0:
                data_unit.append(0.0)
            else:
                temp_increase_per = ((pp_data['open'][temp_index] / start_price) - 1.0) * 100.0
                data_unit.append(temp_increase_per)
    elif label_type == LABEL_CONSECUTIVE_RISE_SCORE:
        max_sum_score = -20.0
        sum_score = 0.0
        day_score = 0.0
        for iloop in range(0, label_days):
            temp_index = start_day_index - iloop
            if temp_index < 0:
                data_unit.append(0.0)
            else:
                # day_score = pp_data['close_increase'][temp_index] - 5.0
                if pp_data['close_increase'][temp_index] > 0.0:
                    day_score = pp_data['close_increase'][temp_index]
                else:
                    day_score = pp_data['close_increase'][temp_index] * 2
                sum_score += day_score
                # if sum_score > max_sum_score:
                #     max_sum_score = sum_score
                # # if (iloop == 8) and ((sum_score > 10) or (sum_score < -10)):
                # #     print('%04u, %f, %s, %s, %f, %f' % (iloop, sum_score, pp_data['ts_code'][temp_index], \
                # #         pp_data['trade_date'][temp_index], \
                # #         pp_data['close'][temp_index], \
                # #         pp_data['close_increase'][temp_index]))
                # data_unit.append(max_sum_score)
                if sum_score < 0.0:
                    sum_score = 0.0
                data_unit.append(sum_score)
    return True

# 从 day_index-1 开始的后 label_days 天的数据，不包含 day_index
def AppendActureData(pp_data, day_index, data_unit):
    temp_str = pp_data['ts_code'][0]
    ts_code_value = float(temp_str[0:6])
    start_day_index = day_index - 1
    for iloop in range(0, label_days):
        temp_index = start_day_index - iloop
        if temp_index < 0:
            data_unit.append(0.0)
            data_unit.append(0.0)
            data_unit.append(0.0)
            data_unit.append(0.0)
            data_unit.append(0.0)
            data_unit.append(ts_code_value)
            data_unit.append(INVALID_DATE)
        else:
            data_unit.append(pp_data['open_increase'][temp_index])
            data_unit.append(pp_data['low_increase'][temp_index])
            data_unit.append(pp_data['open'][temp_index])
            data_unit.append(pp_data['low'][temp_index])
            data_unit.append(pp_data['close'][temp_index])
            temp_str = pp_data['ts_code'][temp_index]
            data_unit.append(float(temp_str[0:6]))
            temp_str = pp_data['trade_date'][temp_index]
            data_unit.append(float(temp_str))
    return True

def GetDataUnit(pp_data, day_index):
    data_unit=[]
    if not AppendFeature(pp_data, day_index, data_unit):
        return []
    if not AppendLabel(pp_data, day_index, data_unit):
        return []
    if not AppendActureData(pp_data, day_index, data_unit):
        return []
    return data_unit

if __name__ == "__main__":
    print('features.py')

