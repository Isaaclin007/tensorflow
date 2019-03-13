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

reload(sys)
sys.setdefaultencoding('utf-8')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

# 配置信息
test_date = '20150131'
filter_days = 300
trading_halt_permit_days = 300
sleep_count_threshold = 400
breakup_count_threshold = 1
code_list, name_list = tushare_data.StockCodesName('20100101', '', '')

def GetDownloadMergeData():
    # 计算需要下载数据的天数，下载数据
    ref_trade_day_num = tushare_data.preprocess_ref_days + \
                        filter_days + \
                        sleep_count_threshold + \
                        breakup_count_threshold + \
                        trading_halt_permit_days
    date_list = tushare_data.TradeDateList(test_date, ref_trade_day_num)

    for date_index in range(0, len(date_list)):
        temp_date = date_list[date_index]
        tushare_data.DownloadATradeDayData(temp_date)
        print("%-4d : %s 100%% download" % (date_index, temp_date))

    # 将每天的数据合并
    # start_flag = True
    # for date_index in range(0, len(date_list)):
    #     temp_date = date_list[date_index]
    #     load_df = tushare_data.LoadATradeDayData(temp_date)
    #     if start_flag:
    #         merge_df = load_df
    #         start_flag = False
    #     else:
    #         merge_df = merge_df.append(load_df)
    #     print("%-4d : %s 100%% merge" % (date_index, temp_date))

    merge_df = pd.DataFrame()
    for date_index in range(0, len(date_list)):
        temp_date = date_list[date_index]
        load_df = tushare_data.LoadATradeDayData(temp_date)
        merge_df = merge_df.append(load_df)
        print("%-4d : %s 100%% merge" % (date_index, temp_date))
    return merge_df

# 生成 preprocessed 集合
file_name = './temp_data/merge_pp_data_%s.csv' % test_date
if os.path.exists(file_name):
    merge_pp_data = pd.read_csv(file_name)
else:
    merge_pp_data = pd.DataFrame()
    download_merge_data = GetDownloadMergeData()
    for code_index in range(0, len(code_list)):
        stock_code = code_list[code_index]
        stock_df = download_merge_data[download_merge_data['ts_code'] == stock_code]
        stock_pp_data = tushare_data.StockDataPreProcess(stock_df)
        merge_pp_data = merge_pp_data.append(stock_pp_data)
        print("%-4d : %s 100%% preprocessed" % (code_index, stock_code))
    merge_pp_data.to_csv(file_name)



# 获取股票列表
result_df = pd.DataFrame()
print("preprocessed data required: %d" % (filter_days + sleep_count_threshold + breakup_count_threshold))
for code_index in range(0, len(code_list)):
    stock_code = code_list[code_index]
    stock_name = name_list[code_index]
    processed_df = merge_pp_data[merge_pp_data['ts_code'] == stock_code]
    if (len(processed_df) >= (filter_days + sleep_count_threshold + breakup_count_threshold)):
        # for day_index in reversed(range(0, filter_days)):
        #     if 1 == breakup_kernel.BreakupStatusAStockADate(processed_df, day_index, sleep_count_threshold, breakup_count_threshold):
        #         trade_date = processed_df.loc[day_index,'trade_date']
        #         print("%12s%16s%12s" % (stock_code, stock_name, trade_date))
        #         row={'1.ts_code':stock_code, '2.name':stock_name, '3.trade_date':trade_date}
        #         result_df = result_df.append(row, ignore_index=True)

        temp_df = breakup_kernel.BreakupHistoryAStock(processed_df, sleep_count_threshold, breakup_count_threshold)
        if not temp_df.empty:
            result_df = result_df.append(temp_df)
            print(result_df)
        print("%-4d : %s 100%%" % (code_index, stock_code))
    else:
        print("%-4d : %s error len(processed_df): %d" % (code_index, stock_code, len(processed_df)))


print(result_df)
result_df.to_csv('./breakup_filter_result.csv')

        


