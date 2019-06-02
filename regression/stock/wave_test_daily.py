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
import wave_kernel
import random
import daily_data
import wave_test_regression

reload(sys)
sys.setdefaultencoding('utf-8')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
pridect_mode = False


# pridect_mode:
# True: 预测与买入信号和预卖出信号，用于实战，只打印预交易信号
# False：回归测试
def TestAllStocksDailyData():
    if not os.path.exists(wave_kernel.FileNameDailyDataSet()):
        result_df = pd.DataFrame()
        increase_sum = 0.0
        holding_days_sum = 0
        trade_count_sum = 0.0
        trade_count_profitable_sum = 0.0
        code_list = tushare_data.StockCodes()
        pp_merge_data = daily_data.GetPreprocessedMergeData()
        for code_index in range(0, len(code_list)):
            stock_code = code_list[code_index]
            pp_data = daily_data.GetPreprocessedData(pp_merge_data, stock_code)
            if len(pp_data) == 0:
                continue
            wave_kernel.AppendWaveData(pp_data)
            if pridect_mode:
                temp_increase, temp_holding_days, trade_count, trade_count_profitable = \
                    wave_kernel.TradeTest(pp_data, 0.05, False, True, True, False, False, True)
            else:
                temp_increase, temp_holding_days, trade_count, trade_count_profitable = \
                    wave_kernel.TradeTest(pp_data, 0.05, False, False, False, False, True, True)
            increase_sum += temp_increase
            holding_days_sum += temp_holding_days
            trade_count_sum += trade_count
            trade_count_profitable_sum += trade_count_profitable
            if holding_days_sum > 0:
                daily_increase = increase_sum/holding_days_sum
            else:
                daily_increase = 0.0
            
            if trade_count_sum > 0:
                avg_increase = increase_sum / trade_count_sum
                profitable_ratio = trade_count_profitable_sum / trade_count_sum
            else:
                avg_increase = 0.0
                profitable_ratio = 0.0
            if not pridect_mode:
                print("%-4d : %s 100%%, %-8.2f, %-8.2f, %-8.2f, %-8.2f, %u/%u:%.2f" % (
                    code_index, \
                    stock_code, \
                    temp_increase, \
                    increase_sum, \
                    avg_increase, \
                    daily_increase, \
                    trade_count_profitable_sum, \
                    trade_count_sum, \
                    profitable_ratio))
            # row = {'ts_code':stock_code, 'increase':temp_increase}
            # result_df = result_df.append(row, ignore_index=True)
        # result_df.to_csv('wave_test_result.csv')
        wave_kernel.SaveDailyDataSet()
    
    dataset = wave_kernel.GetDailyDataSet()
    wave_test_regression.RegressionTest(dataset)

def TestAStocksDailyData(stock_code):
    increase_sum = 0.0
    holding_days_sum = 0
    trade_count_sum = 0.0
    trade_count_profitable_sum = 0.0
    pp_merge_data = daily_data.GetPreprocessedMergeData()
    pp_data = daily_data.GetPreprocessedData(pp_merge_data, stock_code)
    print(pp_data)
    pp_data.to_csv('./temp.csv')
    if len(pp_data) == 0:
        print('len(pp_data) == 0')
        return
    wave_kernel.AppendWaveData(pp_data)
    temp_increase, temp_holding_days, trade_count, trade_count_profitable = \
        wave_kernel.TradeTest(pp_data, 0.05, True, True, True, True, False, False)
    increase_sum += temp_increase
    holding_days_sum += temp_holding_days
    trade_count_sum += trade_count
    trade_count_profitable_sum += trade_count_profitable
    if holding_days_sum > 0:
        daily_increase = increase_sum/holding_days_sum
    else:
        daily_increase = 0.0
    
    if trade_count_sum > 0:
        avg_increase = increase_sum / trade_count_sum
        profitable_ratio = trade_count_profitable_sum / trade_count_sum
    else:
        avg_increase = 0.0
        profitable_ratio = 0.0
    if not pridect_mode:
        print("%-4d : %s 100%%, %-8.2f, %-8.2f, %-8.2f, %-8.2f, %u/%u:%.2f" % (
            0, \
            stock_code, \
            temp_increase, \
            increase_sum, \
            avg_increase, \
            daily_increase, \
            trade_count_profitable_sum, \
            trade_count_sum, \
            profitable_ratio))

if __name__ == "__main__":
    if len(sys.argv) > 1:
        TestAStocksDailyData(sys.argv[1])
    else:
        TestAllStocksDailyData()
    




