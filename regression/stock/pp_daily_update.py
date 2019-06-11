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
import breakup_kernel
import random
import matplotlib.pyplot as plt
import matplotlib as mpl
from datetime import datetime 
import matplotlib.dates as mdate

reload(sys)
sys.setdefaultencoding('utf-8')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

code_list = tushare_data.StockCodes()

def CreatePPMergeDataOriginal():
    merge_file_name = tushare_data.FileNameMergePPDataOriginal()
    print(merge_file_name)
    if not os.path.exists(merge_file_name):
        merge_pp_data = pd.DataFrame()
        for code_index in range(0, len(code_list)):
            stock_code = code_list[code_index]
            file_name = tushare_data.FileNameStockPreprocessedData(stock_code)
            if os.path.exists(file_name):
                stock_pp_data = pd.read_csv(file_name)
                if len(stock_pp_data) >= 200:
                    stock_pp_data = stock_pp_data[:400].copy()
                    merge_pp_data = merge_pp_data.append(stock_pp_data)
                    print("%-4d : %s 100%% merged" % (code_index, stock_code))
        merge_pp_data.to_csv(merge_file_name)

def GetPPMergeDataOriginal():
    merge_file_name = tushare_data.FileNameMergePPDataOriginal()
    merge_pp_data = pd.read_csv(merge_file_name)
    return merge_pp_data

def AddAvg(stock_pp_data, stock_daily_data, target_name, avg_period):
    avg_name = '%s_%u_avg' % (target_name, avg_period)
    sum_value = stock_daily_data.loc[0, target_name]
    for iloop in range(0, avg_period - 1):
        sum_value += stock_pp_data.loc[iloop, target_name]
    stock_daily_data[avg_name] = sum_value / float(avg_period)

g_avg_col_names = [
        'close', 
        'vol'
        ]

def CleanCols(df_data):
    col_captions = df_data.columns.values.tolist()
    for caption in col_captions:
        if len(caption) >= 7:
            if caption[0:7] == 'Unnamed':
                df_data = df_data.drop([caption], axis=1)
    cols_list = ['amount', 'change', 'vol_5', 'pct_chg']
    for drop_col in cols_list:
        if drop_col in col_captions:
            df_data = df_data.drop(drop_col, axis=1)
    return df_data

def Increase(dynamic_value, basic_value):
    return ((dynamic_value / basic_value) - 1.0) * 100.0

def UpdatePPMergeData(merge_pp_data, daily_download_data):
    merge_pp_data = CleanCols(merge_pp_data)
    daily_download_data = CleanCols(daily_download_data)
    update_merge_pp_data = pd.DataFrame()
    for code_index in range(0, len(code_list)):
        ts_code = code_list[code_index]
        stock_pp_data = merge_pp_data[merge_pp_data['ts_code'] == ts_code].copy()
        stock_pp_data = stock_pp_data.sort_values(by=['trade_date'], ascending=(False))
        stock_pp_data = stock_pp_data.reset_index(drop=True)
        if len(stock_pp_data) < 200:
            print('UpdatePPMergeData.Error, len(stock_pp_data):%u' % len(stock_pp_data))
            continue
        stock_pp_data = stock_pp_data[:400].copy()

        # 提取 daily_download_data 中的匹配数据
        add_row_data = daily_download_data[daily_download_data['ts_code'] == ts_code].copy().reset_index(drop=True)
        # add_row_data.reset_index(drop=True)
        if len(add_row_data) > 1:
            print('UpdatePPMergeData.Error, len(add_row_data):%u' % len(add_row_data))
            return
        if len(add_row_data) == 1:
            # 添加 preprocess 数据
            # if add_row_data.loc[0,'pre_close'] != stock_pp_data.loc[0,'close']:
            #     print(stock_pp_data)
            #     print(add_row_data)
            #     print('UpdatePPMergeData.Error, pre_close:%f|%f' % (add_row_data.loc[0,'pre_close'], stock_pp_data.loc[0,'close']))
            #     return
            add_row_data.loc[0,'pre_close'] = stock_pp_data.loc[0,'close']
            add_row_data['open_increase'] = Increase(add_row_data.loc[0,'open'], add_row_data.loc[0,'pre_close'])
            add_row_data['close_increase'] = Increase(add_row_data.loc[0,'close'], add_row_data.loc[0,'pre_close'])
            add_row_data['high_increase'] = Increase(add_row_data.loc[0,'high'], add_row_data.loc[0,'pre_close'])
            add_row_data['low_increase'] = Increase(add_row_data.loc[0,'low'], add_row_data.loc[0,'pre_close'])
            for col_name in g_avg_col_names:
                AddAvg(stock_pp_data, add_row_data, col_name, 5)
                AddAvg(stock_pp_data, add_row_data, col_name, 10)
                AddAvg(stock_pp_data, add_row_data, col_name, 30)
                AddAvg(stock_pp_data, add_row_data, col_name, 100)
                AddAvg(stock_pp_data, add_row_data, col_name, 200)
            add_row_data['close_increase_to_5_avg'] = Increase(add_row_data.loc[0,'close'], add_row_data.loc[0,'close_5_avg'])
            add_row_data['close_increase_to_10_avg'] = Increase(add_row_data.loc[0,'close'], add_row_data.loc[0,'close_10_avg'])
            add_row_data['close_increase_to_30_avg'] = Increase(add_row_data.loc[0,'close'], add_row_data.loc[0,'close_30_avg'])
            add_row_data['close_increase_to_100_avg'] = Increase(add_row_data.loc[0,'close'], add_row_data.loc[0,'close_100_avg'])
            add_row_data['close_increase_to_200_avg'] = Increase(add_row_data.loc[0,'close'], add_row_data.loc[0,'close_200_avg'])
            add_row_data['open_5'] = stock_pp_data.loc[3,'open']
            add_row_data['close_5'] = add_row_data.loc[0,'close']
            high_5 = add_row_data.loc[0,'high']
            low_5 = add_row_data.loc[0,'low']
            for iloop in range(0, 4):
                if high_5 < stock_pp_data.loc[iloop, 'high']:
                    high_5 = stock_pp_data.loc[iloop, 'high']
                if low_5 > stock_pp_data.loc[iloop, 'low']:
                    low_5 = stock_pp_data.loc[iloop, 'low']
            add_row_data['high_5'] = high_5
            add_row_data['low_5'] = low_5
            # print(add_row_data.dtypes)
            # print('\n\n\n')
            # print(stock_pp_data.dtypes)
            # print(stock_pp_data)
            stock_pp_data = add_row_data.append(stock_pp_data, ignore_index=True, sort=False)
        update_merge_pp_data = update_merge_pp_data.append(stock_pp_data, sort=False)
    return update_merge_pp_data


if __name__ == "__main__":
    # CreatePPMergeDataOriginal()
    date_list = tushare_data.TradeDateList(tushare_data.CurrentDate(), 100)
    date_index = 0
    for iloop in range(0, len(date_list)):
        file_name = tushare_data.FileNameMergePPData(date_list[iloop])
        # print(file_name)
        if os.path.exists(file_name):
            date_index = iloop
            break
    if date_index > 0:
        merge_pp_data = pd.read_csv(file_name)
        for iloop in reversed(range(0, date_index)):
            tushare_data.DownloadATradeDayData(date_list[iloop])
            daily_data = tushare_data.LoadATradeDayData(date_list[iloop])
            merge_pp_data = UpdatePPMergeData(merge_pp_data, daily_data)
            print("%-4d : %s 100%% update" % (iloop, date_list[iloop]))
        file_name = tushare_data.FileNameMergePPData(date_list[0])
        print(merge_pp_data)
        merge_pp_data.to_csv(file_name)


        


