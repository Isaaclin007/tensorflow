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

reload(sys)
sys.setdefaultencoding('utf-8')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

PERIOD_STATUS_INIT = 0
PERIOD_STATUS_UP = 1
PERIOD_STATUS_DOWN = 2
PERIOD_STATUS_FINISH = 3

# 分析一支股票的一个时间点的状态
# 返回值：1=breakup触发状态，2=breakup保持状态，0=其他状态
def BreakupStatusAStockADate(input_pp_data, \
                      intput_day_index, \
                      input_sleep_count_threshold, \
                      input_breakup_count_threshold):
    period_status = PERIOD_STATUS_INIT
    period_continue_up_count = 0
    period_continue_sleep_count = 0
    for day_loop in range(intput_day_index, len(input_pp_data)):
        in_target = input_pp_data.loc[day_loop,'close_10_avg']
        close_avg = input_pp_data.loc[day_loop,'close_100_avg']
        if in_target > close_avg:
            # 状态跳转
            if period_status == PERIOD_STATUS_INIT:
                period_status = PERIOD_STATUS_UP
            elif period_status == PERIOD_STATUS_DOWN:
                period_status = PERIOD_STATUS_FINISH
            #更新计数器
            if period_status == PERIOD_STATUS_UP:
                period_continue_up_count += 1
        else:
            # 状态跳转
            if (period_status == PERIOD_STATUS_INIT) or (period_status == PERIOD_STATUS_UP):
                period_status = PERIOD_STATUS_DOWN
            #更新计数器
            if period_status == PERIOD_STATUS_DOWN:
                period_continue_sleep_count += 1
    
    if period_continue_sleep_count >= input_sleep_count_threshold:
        if period_continue_up_count == input_breakup_count_threshold:
            return 1
        elif period_continue_up_count > input_breakup_count_threshold:
            return 2
    return 0



# 分析一支股票的历史状态
# only_output_event: True=值返回breakup事件, False=返回breakup状态
# 返回值：breakup 事件列表 pandas dataframe
def BreakupHistoryAStock(input_pp_data, \
                        input_sleep_count_threshold, \
                        input_breakup_count_threshold, \
                        only_output_event):
    period_status = PERIOD_STATUS_INIT
    period_continue_sleep_count = 0
    period_continue_up_count = 0
    result_df = pd.DataFrame()
    for day_loop in reversed(range(0, len(input_pp_data))):
        target = input_pp_data.loc[day_loop,'close_10_avg']
        close_avg = input_pp_data.loc[day_loop,'close_100_avg']

        if target > close_avg:
            if period_status != PERIOD_STATUS_UP:
                period_status = PERIOD_STATUS_UP
                period_continue_up_count = 0
            period_continue_up_count += 1
            if period_continue_sleep_count >= input_sleep_count_threshold:
                if ((not only_output_event) and (period_continue_up_count >= input_breakup_count_threshold)) \
                      or (only_output_event and (period_continue_up_count == input_breakup_count_threshold)):
                    stock_code = input_pp_data.loc[day_loop, 'ts_code']
                    trade_date = input_pp_data.loc[day_loop, 'trade_date'].astype(str)
                    row = {'ts_code':stock_code, 'trade_date':trade_date, 'sleep_count':period_continue_sleep_count}
                    result_df = result_df.append(row, ignore_index=True)
        else:
            if period_status != PERIOD_STATUS_DOWN:
                period_status = PERIOD_STATUS_DOWN
                period_continue_sleep_count = 0
            period_continue_sleep_count += 1
    return result_df


