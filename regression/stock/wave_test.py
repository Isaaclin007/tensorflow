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

reload(sys)
sys.setdefaultencoding('utf-8')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

def GetProprocessedData(ts_code):
    stock_pp_file_name = tushare_data.FileNameStockPreprocessedData(ts_code)
    if os.path.exists(stock_pp_file_name):
        pp_data = pd.read_csv(stock_pp_file_name)
        pp_data = pp_data[pp_data['trade_date'] >= wave_kernel.start_date].copy().reset_index(drop=True)
        # print(pp_data)
        return pp_data
    else:
        return pd.DataFrame()

def TestAStockLowLevel(ts_code, \
                       print_finished_record, \
                       print_unfinished_record, \
                       print_trade_flag, \
                       print_summary):
    # tushare_data.DownloadAStocksData(ts_code)
    # tushare_data.UpdatePreprocessDataAStock(-1, ts_code)
    pp_data = GetProprocessedData(ts_code)
    if len(pp_data) == 0:
        return 0.0, 0, 0, 0.0
    wave_kernel.AppendWaveData(pp_data)
    # pp_data = pp_data[pp_data['trade_date'] >= 20180101]
    # pp_data = pp_data[pp_data['trade_date'] < 20190101]
    # pp_data = pp_data[pp_data['trade_date'] < 20180101]
    if len(pp_data) == 0:
        return 0.0, 0, 0, 0.0
    pp_data = pp_data.copy()
    pp_data = pp_data.reset_index(drop=True)
    return wave_kernel.TradeTest(pp_data, \
                                 0.05, \
                                 print_finished_record, \
                                 print_unfinished_record, \
                                 print_trade_flag, \
                                 print_summary, 
                                 True,
                                 True)

def TestAStock(ts_code):
    return TestAStockLowLevel(ts_code, True, True, True, True)

def TestAllStocks():
    result_df = pd.DataFrame()
    increase_sum = 0.0
    holding_days_sum = 0
    trade_count_sum = 0.0
    trade_count_profitable_sum = 0.0
    code_list = tushare_data.StockCodes()
    for code_index in range(0, len(code_list)):
        stock_code = code_list[code_index]
        temp_increase, temp_holding_days, trade_count, trade_count_profitable = \
            TestAStockLowLevel(stock_code, False, False, False, False)
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
        else:
            avg_increase = 0.0
        if trade_count_sum > 0:
            profitable_ratio = trade_count_profitable_sum / trade_count_sum
        else:
            profitable_ratio = 0.0
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
        row = {'ts_code':stock_code, 'increase':temp_increase}
        result_df = result_df.append(row, ignore_index=True)
    # print("%-4d : %s 100%%, %-8.2f, %-8.2f, %-8.2f, %-8.2f, %u/%u:%.2f" % (
    #     code_index, \
    #     stock_code, \
    #     temp_increase, \
    #     increase_sum, \
    #     avg_increase, \
    #     daily_increase, \
    #     trade_count_profitable_sum, \
    #     trade_count_sum, \
    #     profitable_ratio))
    result_df.to_csv('wave_test_result.csv')
    # wave_kernel.SaveTrainData()
    wave_kernel.SaveDataSet()


if __name__ == "__main__":
    # TestAStock('600104.SH')
    # TestAStock(sys.argv[1])
    # TestAllStocks()
    if len(sys.argv) > 1:
        TestAStock(sys.argv[1])
    else:
        TestAllStocks()




