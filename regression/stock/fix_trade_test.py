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

def GetDownloadData(ts_code):
    stock_file_name = tushare_data.FileNameStockDownloadDataDaily(ts_code)
    if os.path.exists(stock_file_name):
        df_data = pd.read_csv(stock_file_name)
        return df_data
    else:
        return pd.DataFrame()

def TestAStock(ts_code, show_trade_record):
    df_data = GetDownloadData(ts_code)
    if len(df_data) == 0:
        return 0.0
    current_mon = 0
    vol_sum = 0.0
    amount_sum = 0.0
    avg_increase = 0.0
    avg_cost = 0.0
    for day_index in reversed(range(0, len(df_data))):
        temp_mon = int(df_data.loc[day_index, 'trade_date']) / 100
        if temp_mon >= 201001 and current_mon != temp_mon:
            if avg_increase < -20:
                temp_amount = 40000.0
            elif avg_increase < -10:
                temp_amount = 20000.0
            else:
                temp_amount = 10000.0
            current_mon = temp_mon
            current_price = df_data.loc[day_index, 'open']
            amount_sum += temp_amount
            vol_sum += temp_amount / current_price
            avg_cost = amount_sum / vol_sum
            avg_increase = (current_price - avg_cost) / avg_cost * 100.0
            if show_trade_record:
                print("%-10s%-10.2f%-10.0f%-10.0f%-10.2f%-10.2f" % ( \
                    df_data.loc[day_index, 'trade_date'], \
                    current_price, \
                    amount_sum, \
                    vol_sum, \
                    avg_cost, \
                    avg_increase))
    return current_price * vol_sum, amount_sum



def TestAllStocks():
    result_df = pd.DataFrame()
    amount_sum = 0.0
    cost_sum = 0.0
    code_list = tushare_data.StockCodes()
    for code_index in range(0, len(code_list)):
        stock_code = code_list[code_index]
        temp_amount, temp_cost = TestAStock(stock_code, False)
        amount_sum += temp_amount
        cost_sum += temp_cost
        avg_increase = (amount_sum - cost_sum) / cost_sum * 100.0
        print("%-4d : %s 100%%, %-8.2f, %-8.2f, %-8.2f" % (
            code_index, \
            stock_code, \
            amount_sum, \
            cost_sum, \
            avg_increase))
        temp_increase = (temp_amount - temp_cost) / temp_cost * 100.0
        row = {'ts_code':stock_code, 'increase':temp_increase}
        result_df = result_df.append(row, ignore_index=True)
    result_df.to_csv('fix_trade_test_result.csv')



if __name__ == "__main__":
    if len(sys.argv) > 1:
        TestAStock(sys.argv[1], True)
    else:
        TestAllStocks()




