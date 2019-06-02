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

# 配置信息
start_date = '20180101'
end_date = '20190411'
# end_date = tushare_data.CurrentDate()
code_list = tushare_data.StockCodes()


def GetDownloadMergeData():
    file_name = './data/temp/daily_data_merge_download_data_%s_%s.csv' % (start_date, end_date)
    if os.path.exists(file_name):
        merge_df = pd.read_csv(file_name)
    else:
        # 计算需要下载数据的天数，下载数据
        date_list = tushare_data.TradeDateListRange(start_date, end_date)
        for date_index in range(0, len(date_list)):
            temp_date = date_list[date_index]
            tushare_data.DownloadATradeDayData(temp_date)
            print("%-4d : %s 100%% download" % (date_index, temp_date))

        merge_df = pd.DataFrame()
        for date_index in range(0, len(date_list)):
            temp_date = date_list[date_index]
            load_df = tushare_data.LoadATradeDayData(temp_date)
            merge_df = merge_df.append(load_df)
            print("%-4d : %s 100%% merge" % (date_index, temp_date))
        merge_df.to_csv(file_name)
    return merge_df

# 生成 preprocessed 集合
def GetPreprocessedMergeData():
    file_name = './data/temp/daily_data_merge_pp_data_%s_%s.csv' % \
                (start_date, \
                end_date)
    if os.path.exists(file_name):
        merge_pp_data = pd.read_csv(file_name)
    else:
        merge_pp_data = pd.DataFrame()
        download_merge_data = GetDownloadMergeData()
        for code_index in range(0, len(code_list)):
            stock_code = code_list[code_index]
            stock_df = download_merge_data[download_merge_data['ts_code'] == stock_code]
            stock_pp_data = tushare_data.StockDataPreProcess(stock_df, False)
            if len(stock_pp_data) > 0:
                merge_pp_data = merge_pp_data.append(stock_pp_data)
                print("%-4d : %s 100%% preprocessed" % (code_index, stock_code))
        merge_pp_data.to_csv(file_name)
    return merge_pp_data

def GetDownloadData(merge_data, ts_code):
    temp_df = merge_data[merge_data['ts_code'] == ts_code]
    pp_copy = temp_df.copy()
    pp_copy = pp_copy.reset_index(drop=True)
    return pp_copy

def GetPreprocessedData(merge_pp_data, ts_code):
    processed_df = merge_pp_data[merge_pp_data['ts_code'] == ts_code]
    pp_data_copy = processed_df.copy()
    pp_data_copy = pp_data_copy.reset_index(drop=True)
    return pp_data_copy


if __name__ == "__main__":
    df = GetDownloadMergeData()
    stock_df = GetDownloadData(df, '002070.SZ')
    # df = GetPreprocessedMergeData()
    # stock_df = GetPreprocessedData(df, '002070.SZ')
    print(stock_df)
    stock_df.to_csv('./temp.csv')
        


