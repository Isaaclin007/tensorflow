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

min_wave_width_left = 4
min_wave_width_right = 4
trade_off_threshold = 0
pe_ttm_threshold = 10
wave_index = 'close'

EXTREME_NONE = 0
EXTREME_PEAK = 1
EXTREME_VALLEY = 2

STATUS_NONE = 0
STATUS_UP = 1
STATUS_DOWN = 2

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

def TradeTest(input_pp_data, \
              cut_loss_ratio, \
              print_finished_record, \
              print_unfinished_record, \
              print_trade_flag, \
              print_summary):
    if len(input_pp_data) == 0:
        return 0.0, 0
    ts_code = input_pp_data.loc[0, 'ts_code']
    input_pp_data['wave_trade'] = TRADE_NONE
    data_len = len(input_pp_data)
    current_trade_status = TRADE_OFF
    last_peak = -1.0
    last_valley = -1.0
    day_index = data_len - 1
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
        # close_avg = input_pp_data.loc[day_index, 'close_100_avg']
        wave_extreme = input_pp_data.loc[day_index, 'wave_extreme']
        wave_status = input_pp_data.loc[day_index, 'wave_status']
        trade_flag = TRADE_NONE
        if last_peak > 0.0 and last_valley > 0.0:
            if current_trade_status == TRADE_OFF:
                # ON 事件产生
                # if (wave_status == STATUS_UP or wave_status == STATUS_NONE) and \
                if close > last_peak and \
                   trade_off_count > trade_off_threshold:
                    # print('%s,wave_status == STATUS_UP and close > last_peak:%f>%f' % (current_date, close, last_peak))
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
                    # print('TRADE_ON, %s offset:%u' % (current_date, trade_day_offset))
                    if day_index >= trade_day_offset:
                        day_index -= trade_day_offset
                        on_price = input_pp_data.loc[day_index, 'open']
                        current_trade_status = TRADE_ON
                        on_date = input_pp_data.loc[day_index,'trade_date']
                        temp_holding_days = 0
                    else:
                        # 打印预买入信号
                        if print_trade_flag:
                            print("%-6u%-10s%-12s%-12s%-12s%-10u%-10s%-10s%-10s%-8u%-8s" %( \
                                trade_count, \
                                ts_code, \
                                '%s+%u' % (current_date, trade_day_offset), \
                                '--', \
                                '--', \
                                temp_holding_days, \
                                '--', \
                                '--', \
                                '--', \
                                on_reason, \
                                '--'))
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
                temp_holding_days += 1
                # OFF 事件处理
                if trade_flag == TRADE_OFF:
                    if day_index >= trade_day_offset:
                        day_index -= trade_day_offset
                        off_price = input_pp_data.loc[day_index, 'open']
                        current_trade_status = TRADE_OFF
                        off_date = input_pp_data.loc[day_index,'trade_date']
                        temp_holding_days += trade_day_offset
                        # 计算本机交易收益
                        temp_increase = ((off_price / on_price) - 1.0) * 100.0
                        sum_increase += temp_increase
                        sum_holding_days += temp_holding_days
                        # 打印交易记录
                        if print_finished_record:
                            print("%-6u%-10s%-12s%-12s%-12s%-10u%-10.2f%-10.2f%-10.2f%-8u%-8u" %( \
                            trade_count, \
                            ts_code, \
                            on_date, \
                            current_date, \
                            off_date, \
                            temp_holding_days, \
                            on_price, \
                            off_price, \
                            temp_increase, \
                            on_reason, \
                            off_reason))
                        trade_count += 1
                        if temp_increase > 0:
                            trade_count_profitable += 1
                    else:
                        # 打印预卖出信号
                        if print_trade_flag:
                            print("%-6u%-10s%-12s%-12s%-12s%-10u%-10.2f%-10s%-10s%-8u%-8s" %( \
                                trade_count, \
                                ts_code, \
                                on_date, \
                                current_date, \
                                '%s+%u' % (current_date, trade_day_offset), \
                                temp_holding_days, \
                                on_price, \
                                '--', \
                                '--', \
                                on_reason, \
                                '--'))
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
        # 打印还没有结束的交易
        if print_unfinished_record:
            print("%-6u%-10s%-12s%-12s%-12s%-10u%-10.2f%-10s%-10s%-8u%-8s" %( \
            trade_count, \
            ts_code, \
            on_date, \
            '--', \
            '--', \
            temp_holding_days, \
            on_price, \
            '--', \
            '--', \
            on_reason, \
            '--'))
    if print_summary:
        print("test_days: %u, holding_days_sum: %u, increase: %.2f" % (test_days, sum_holding_days, sum_increase))
    return sum_increase, sum_holding_days, trade_count, trade_count_profitable
    

