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
# test_date = '20160131'
test_date = '20190427'
filter_days = 100
trading_halt_permit_days = 20
sleep_count_threshold = 1
breakup_count_threshold = 1
stocks_list_end_date = '20100101'
code_list, name_list = tushare_data.StockCodesName(stocks_list_end_date, '', '')

ref_trade_day_num = tushare_data.preprocess_ref_days + \
                    filter_days + \
                    sleep_count_threshold + \
                    breakup_count_threshold + \
                    trading_halt_permit_days
date_list = tushare_data.TradeDateList(test_date, ref_trade_day_num)

def GetDownloadMergeData():
    # 计算需要下载数据的天数，下载数据

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
file_name = './data/temp/breakup_filter_merge_pp_data_%s_%s_%u_%u_%u_%u.csv' % \
            (test_date, \
            stocks_list_end_date, \
            filter_days, \
            trading_halt_permit_days, \
            sleep_count_threshold, \
            breakup_count_threshold)
if os.path.exists(file_name):
    merge_pp_data = pd.read_csv(file_name)
else:
    merge_pp_data = pd.DataFrame()
    download_merge_data = GetDownloadMergeData()
    for code_index in range(0, len(code_list)):
        stock_code = code_list[code_index]
        stock_df = download_merge_data[download_merge_data['ts_code'] == stock_code]
        stock_pp_data = tushare_data.StockDataPreProcess(stock_df, False)
        merge_pp_data = merge_pp_data.append(stock_pp_data)
        print("%-4d : %s 100%% preprocessed" % (code_index, stock_code))
    merge_pp_data.to_csv(file_name)


# #### breakup 事件集合
# result_df = pd.DataFrame()
# print("preprocessed data required: %d" % (filter_days + sleep_count_threshold + breakup_count_threshold))
# for code_index in range(0, len(code_list)):
#     stock_code = code_list[code_index]
#     stock_name = name_list[code_index]
#     processed_df = merge_pp_data[merge_pp_data['ts_code'] == stock_code]
#     if (len(processed_df) >= (filter_days + sleep_count_threshold + breakup_count_threshold)):
#         pp_data_copy=processed_df.copy()
#         pp_data_copy=pp_data_copy.reset_index(drop=True)
#         temp_df = breakup_kernel.BreakupHistoryAStock(pp_data_copy, sleep_count_threshold, breakup_count_threshold, True)
#         if not temp_df.empty:
#             result_df = result_df.append(temp_df)
#             print(result_df)
#         print("%-4d : %s 100%%" % (code_index, stock_code))
#     else:
#         print("%-4d : %s error len(processed_df): %d" % (code_index, stock_code, len(processed_df)))

# print(result_df)
# result_df.to_csv('./breakup_filter_result.csv')


# #### breakup 状态集合
def GetBreakupStatusDataSet():
    temp_result_df = pd.DataFrame()
    print("preprocessed data required: %d" % (filter_days + sleep_count_threshold + breakup_count_threshold))
    for code_index in range(0, len(code_list)):
        stock_code = code_list[code_index]
        processed_df = merge_pp_data[merge_pp_data['ts_code'] == stock_code]
        if (len(processed_df) >= (filter_days + sleep_count_threshold + breakup_count_threshold)):
            pp_data_copy=processed_df.copy()
            pp_data_copy=pp_data_copy.reset_index(drop=True)
            temp_df = breakup_kernel.BreakupHistoryAStock(pp_data_copy, sleep_count_threshold, breakup_count_threshold, False)
            if not temp_df.empty:
                temp_result_df = temp_result_df.append(temp_df)
            print("%-4d : %s 100%%" % (code_index, stock_code))
        else:
            print("%-4d : %s error len(processed_df): %d" % (code_index, stock_code, len(processed_df)))
    return temp_result_df

def GetBreakupEventDataSet():
    temp_result_df = pd.DataFrame()
    print("preprocessed data required: %d" % (filter_days + sleep_count_threshold + breakup_count_threshold))
    for code_index in range(0, len(code_list)):
        stock_code = code_list[code_index]
        processed_df = merge_pp_data[merge_pp_data['ts_code'] == stock_code]
        if (len(processed_df) >= (filter_days + sleep_count_threshold + breakup_count_threshold)):
            pp_data_copy=processed_df.copy()
            pp_data_copy=pp_data_copy.reset_index(drop=True)
            temp_df = breakup_kernel.BreakupHistoryAStock(pp_data_copy, sleep_count_threshold, breakup_count_threshold, True)
            if not temp_df.empty:
                temp_result_df = temp_result_df.append(temp_df)
            print("%-4d : %s 100%%" % (code_index, stock_code))
        else:
            print("%-4d : %s error len(processed_df): %d" % (code_index, stock_code, len(processed_df)))
    return temp_result_df

file_name = './data/temp/breakup_filter_result_%s_%s_%u_%u_%u_%u.csv' % \
            (test_date, \
            stocks_list_end_date, \
            filter_days, \
            trading_halt_permit_days, \
            sleep_count_threshold, \
            breakup_count_threshold)
if os.path.exists(file_name):
    result_df = pd.read_csv(file_name)
else:
    result_df = GetBreakupStatusDataSet()
    result_df.to_csv(file_name)

result_df['trade_date'] = result_df['trade_date'].astype(str)
date_breakup_count_df = pd.DataFrame()
for date_index in range(0, len(date_list)):
    temp_date = date_list[date_index]
    temp_date_history_df = result_df[result_df['trade_date'] == temp_date]
    temp_date_breakup_count = len(temp_date_history_df)
    row = {'trade_date':temp_date, 'breakup_count':temp_date_breakup_count}
    date_breakup_count_df = date_breakup_count_df.append(row, ignore_index=True)
fig1 = plt.figure(dpi=70,figsize=(32,10))
ax1 = fig1.add_subplot(1,1,1) 
ax1.xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m-%d'))
plt.title('breakup_count')
plt.xlabel('date')
plt.ylabel('breakup')
xs = [datetime.strptime(d, '%Y%m%d').date() for d in date_list]
plt.grid(True)
plt.plot(xs, date_breakup_count_df['breakup_count'].values, label='count', linewidth=1)
plt.gcf().autofmt_xdate()
plt.legend()
plt.show()

file_name = './data/temp/breakup_event_result_%s_%s_%u_%u_%u_%u.csv' % \
            (test_date, \
            stocks_list_end_date, \
            filter_days, \
            trading_halt_permit_days, \
            sleep_count_threshold, \
            breakup_count_threshold)
if not os.path.exists(file_name):
    result_df = GetBreakupEventDataSet()
    result_df.to_csv(file_name)

        


