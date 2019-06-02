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

test_date_list = tushare_data.TestTradeDateList()
# test_date_list = test_date_list[0:100]
# print("len(test_date_list):")
# print(len(test_date_list))
# load_data = np.load('./temp_data/test_data_20090101_20190104_100_1_1_0.npy')

def GetProprocessedData(ts_code):
    stock_pp_file_name = tushare_data.FileNameStockPreprocessedData(ts_code)
    if os.path.exists(stock_pp_file_name):
        pp_data = pd.read_csv(stock_pp_file_name)
        return pp_data

def TestAStockLowLevel(ts_code, print_record):
    tushare_data.DownloadAStocksData(ts_code)
    tushare_data.UpdatePreprocessDataAStock(-1, ts_code)
    pp_data = GetProprocessedData(ts_code)
    # pp_data = pp_data[pp_data['trade_date'] >= 20150101]
    # pp_data = pp_data[pp_data['trade_date'] < 20160101]

    pp_data = pp_data.copy()
    pp_data = pp_data.reset_index(drop=True)
    sleep_count = 0
    period_trade_count = 0
    sleep_count_threshold = 5
    breakup_count_threshold = 0
    avg_up_continue_count = 0
    break_up = False
    holding = False
    trade_count = 0
    increase_sum = 0.0
    holding_days_sum = 0
    holding_days = 0
    test_days = 0
    out_target_trough = 0.0
    out_target_trough_pre = 0.0
    out_target_peak = 0.0
    out_target_max = 0.0
    out_target_trend = 1
    out_target_trend_pre = 1
    out_target_diff = 0.0
    break_up_price = 0.0
    capital_value = 1.0
    max_close = 0.0
    for day_loop in reversed(range(0, len(pp_data) - 1)):
        close = pp_data.loc[day_loop,'close']
        in_target = pp_data.loc[day_loop,'close_10_avg']
        out_target = pp_data.loc[day_loop,'close_10_avg']
        out_target_pre = pp_data.loc[day_loop+1,'close']
        close_avg = pp_data.loc[day_loop,'close_100_avg']
        out_target_diff = out_target - out_target_pre
        if out_target_diff > 0.0:
            out_target_trend_pre = out_target_trend
            out_target_trend = 1
        elif out_target_diff < 0.0:
            out_target_trend_pre = out_target_trend
            out_target_trend = -1
        if (out_target_trend > 0) and (out_target_trend_pre < 0):
            out_target_peak = out_target_pre
        elif (out_target_trend < 0) and (out_target_trend_pre > 0):
            out_target_trough_pre = out_target_trough
            out_target_trough = out_target_pre

        if in_target > close_avg:
            avg_up_continue_count += 1
            if avg_up_continue_count >= breakup_count_threshold:
                break_up = True
        else:
            if break_up:
                sleep_count = 0
            sleep_count += 1
            break_up = False
            period_trade_count = 0
            max_close = 0
        
        if not holding:
            if (sleep_count >= sleep_count_threshold) and break_up and (day_loop > 1) and (close > max_close):
                if (period_trade_count == 0) or ((period_trade_count > 0) and (close > break_up_price)):
                    in_trade_date = pp_data.loc[day_loop - 1,'trade_date']
                    buying_price = pp_data.loc[day_loop - 1,'open']
                    holding = True
                    holding_days = 1
                    out_target_trough = out_target
                    out_target_peak = 0.0
                    max_close = buying_price
                if period_trade_count == 0:
                    break_up_price = buying_price
                period_trade_count += 1
        else:
            # 卖出条件：当out_taget低于最近波谷时卖出，或者最近波峰小于上一个波峰
            # if (not break_up) or (out_target < out_target_peak) or (out_target_trough < out_target_trough_pre):

            # 卖出条件：10% 止损
            # if (not break_up) or (((close - buying_price)/buying_price) < (-0.10)):

            # 卖出条件：target 跌破 avg
            # if (not break_up):

            # 卖出条件：target 跌破 break_up_price
            if max_close < close:
                max_close = close
            if (not break_up) or (close < break_up_price) or (close < (max_close * 0.8)) or (day_loop == 1):
            # if (close < (max_close * 0.9)) or (close < (buying_price * 0.95)) or (day_loop == 1):
                out_trade_date = pp_data.loc[day_loop - 1,'trade_date']
                out_price = pp_data.loc[day_loop - 1,'open']
                holding = False
                trade_count += 1
                temp_increase = (out_price/buying_price - 1.0) * 100.0
                capital_value *= (out_price/buying_price)
                if print_record:
                    print("%6u%10s%10s%10s%10u%10.2f%10.2f%10.2f" %( \
                        trade_count, \
                        ts_code, \
                        in_trade_date, \
                        out_trade_date, \
                        holding_days, \
                        buying_price, \
                        out_price, \
                        temp_increase))
                increase_sum += temp_increase
                holding_days_sum += holding_days
            else:
                holding_days += 1
        test_days += 1

    compound_inclrease = (capital_value - 1.0)*100.0
    if print_record:
        print("test_days: %u, holding_days_sum: %u, increase: %.2f" % (test_days, holding_days_sum, compound_inclrease))
        # print('%8.2f%8.2f%10u%10u%10u%10u' % (\
        #     close, \
        #     close_30_avg, \
        #     sleep_count, \
        #     avg_30_up_continue_count, \
        #     break_up, \
        #     holding))
    # return compound_inclrease
    return increase_sum, holding_days_sum

def TestAStock(ts_code):
    return TestAStockLowLevel(ts_code, True)

def TestAllStocks():
    increase_sum = 0.0
    holding_days_sum = 0
    code_list = tushare_data.StockCodes()
    for code_index in range(0, len(code_list)):
        stock_code = code_list[code_index]
        temp_increase, temp_holding_days = TestAStockLowLevel(stock_code, False)
        increase_sum += temp_increase
        holding_days_sum += temp_holding_days
        if holding_days_sum > 0:
            daily_increase = increase_sum/holding_days_sum
        else:
            daily_increase = 0.0
        print("%-4d : %s 100%%, %.2f, %.2f, %.2f" % (code_index, stock_code, temp_increase, increase_sum, daily_increase))

if __name__ == "__main__":
    # TestAStock('600104.SH')
    # TestAStock(sys.argv[1])
    if len(sys.argv) > 1:
        TestAStock(sys.argv[1])
    else:
        # TestAllStocksDailyData(False)
        TestAllStocks()


