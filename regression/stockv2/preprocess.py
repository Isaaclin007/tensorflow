# -*- coding:UTF-8 -*-


import numpy as np
import pandas as pd
import os
import time
import datetime
import sys
import math
import tushare_data
sys.path.append("..")
from common import base_common
from common import np_common
from common.const_def import *

preprocess_ref_days = 100

reload(sys)
sys.setdefaultencoding('utf-8')

g_trade_date_list = []

def StockDataPreProcess_AddAvg(src_df, target_name, avg_period):
    avg_name = '%s_%u_avg' % (target_name, avg_period)
    src_df[avg_name]=0.0
    current_value = 0.0
    avg_count = 0
    avg_sum = 0.0
    for day_loop in reversed(range(0, len(src_df))):
        current_value = src_df.loc[day_loop, target_name]

        if avg_count < avg_period:
            avg_sum += current_value
            avg_count += 1
        else:
            avg_sum = avg_sum + current_value - src_df.loc[day_loop + avg_period, target_name]
            src_df.loc[day_loop, avg_name] = avg_sum / avg_period

SUSPEND_BORDER_NONE = 0
SUSPEND_BORDER_SUSPEND = 1
SUSPEND_BORDER_RESUMPTION = 2
def StockDataPreProcess_AddSuspendBorder(pp_data, previous_data=[]):
    global g_trade_date_list
    if len(g_trade_date_list) == 0:
        g_trade_date_list = tushare_data.TradeDateLowLevel(base_common.CurrentDate()).astype(np.int64)
    else:
        if int(g_trade_date_list[0]) < int(pp_data[0][PPI_trade_date]):
            print("Error: StockDataPreProcess_AddSuspendBorder, len(src_df)==0")
            return

    src_date_list = pp_data[:, PPI_trade_date]
    dst_index = 0

    for iloop in range(0, len(src_date_list)):
        sync_flag = (g_trade_date_list[dst_index] == src_date_list[iloop])
        if not sync_flag:
            if iloop > 0:
                pp_data[iloop][PPI_suspend] = SUSPEND_BORDER_SUSPEND
                pp_data[iloop - 1][PPI_suspend] = SUSPEND_BORDER_RESUMPTION
            dst_index = int(np.where(g_trade_date_list == src_date_list[iloop])[0])
        dst_index += 1
    if len(previous_data) > 0:
        sync_flag = (g_trade_date_list[dst_index] == int(previous_data[0][PPI_trade_date]))
        if not sync_flag:
            previous_data[0][PPI_suspend] = SUSPEND_BORDER_SUSPEND
            pp_data[len(pp_data)-1][PPI_suspend] = SUSPEND_BORDER_RESUMPTION


def StockDataPreProcess_AddAdjFlag(src_df, previous_df=[]):
    if len(src_df) == 0:
        print("Warning: StockDataPreProcess_AddAdjFlag, len(src_df)==0")
        return
    src_df['adj_flag'] = 0

    current_adj_factor = src_df.loc[len(src_df)-1, 'adj_factor']
    for iloop in reversed(range(0, len(src_df))):
        if current_adj_factor != src_df.loc[iloop, 'adj_factor']:
            src_df.loc[iloop, 'adj_flag'] = 1
            current_adj_factor = src_df.loc[iloop, 'adj_factor']
    if len(previous_df) > 0:
        if src_df.loc[len(src_df)-1, 'adj_factor'] != previous_df.loc[0, 'adj_factor']:
            src_df.loc[len(src_df)-1, 'adj_flag'] = 1

def PPSpread(pp_data, col_index, sum_days):
    sum_len = len(pp_data) - (sum_days - 1)
    spread_data = np.zeros((sum_len, sum_days))
    for iloop in range(sum_days):
        spread_data[:, iloop] = pp_data[iloop:iloop+sum_len, col_index]
    return spread_data

def StockDataPreProcess(df, adj_mode):
    data_len = len(df)
    if data_len <= preprocess_ref_days:
        print('Error:StockDataPreProcess:data_len=%u' % data_len)
        return []

    # get data use flag 
    use_daily_basic = df.columns.contains('turnover_rate_f')
    use_money_flow = df.columns.contains('buy_sm_vol')
    use_adj_factor = df.columns.contains('adj_factor')
    if (adj_mode != '') and (not use_adj_factor):
        print('Error:StockDataPreProcess:adj_mode=%s,use_adj_factor=%u' % (adj_mode, int(use_adj_factor)))
        return []

    # copy df data to np data
    ts_code = df.loc[0, 'ts_code']
    pp_data = np.zeros((data_len, PPI_NUM))
    pp_data[:,PPI_ts_code] = int(ts_code[:6])
    pp_data[:,PPI_trade_date] = df['trade_date'].values
    pp_data[:,PPI_open] = df['open'].values
    pp_data[:,PPI_close] = df['close'].values
    pp_data[:,PPI_high] = df['high'].values
    pp_data[:,PPI_low] = df['low'].values
    pp_data[:,PPI_vol] = df['vol'].values
    if use_daily_basic:
        pp_data[:,PPI_turnover_rate_f] = df['turnover_rate_f'].values
    if use_money_flow:
        pp_data[:,PPI_buy_sm_vol] = df['buy_sm_vol'].values
        pp_data[:,PPI_sell_sm_vol] = df['sell_sm_vol'].values
        pp_data[:,PPI_buy_md_vol] = df['buy_md_vol'].values
        pp_data[:,PPI_sell_md_vol] = df['sell_md_vol'].values
        pp_data[:,PPI_buy_lg_vol] = df['buy_lg_vol'].values
        pp_data[:,PPI_sell_lg_vol] = df['sell_lg_vol'].values
        pp_data[:,PPI_buy_elg_vol] = df['buy_elg_vol'].values
        pp_data[:,PPI_sell_elg_vol] = df['sell_elg_vol'].values
        pp_data[:,PPI_net_mf_vol] = df['net_mf_vol'].values
    if use_adj_factor:
        pp_data[:,PPI_adj_factor] = df['adj_factor'].values

    # verify
    if np.isnan(pp_data).sum() > 0:
        print('Error: StockDataPreProcess invalid data nan')
        return []
    if np.isinf(pp_data).sum() > 0:
        print('Error: StockDataPreProcess invalid data inf')
        return []
    if ((pp_data[:,PPI_trade_date] <= 0).sum() > 0) or \
            ((pp_data[:,PPI_open] <= 0).sum() > 0) or \
            ((pp_data[:,PPI_close] <= 0).sum() > 0) or \
            ((pp_data[:,PPI_high] <= 0).sum() > 0) or \
            ((pp_data[:,PPI_low] <= 0).sum() > 0):
        print('Error: StockDataPreProcess invalid data 0')
        return []

    # adj forward
    if adj_mode == 'f':  # forward
        pp_data[:,PPI_open] *= pp_data[:,PPI_adj_factor]
        pp_data[:,PPI_close] *= pp_data[:,PPI_adj_factor]
        pp_data[:,PPI_high] *= pp_data[:,PPI_adj_factor]
        pp_data[:,PPI_low] *= pp_data[:,PPI_adj_factor]
    
    # pre_close
    pp_data[:data_len-1,PPI_pre_close] = pp_data[1:data_len,PPI_close]
    pp_data[data_len-1][PPI_pre_close] = pp_data[data_len-1][PPI_close]

    # increase
    pp_data[:,PPI_open_increase] = (pp_data[:,PPI_open] / pp_data[:,PPI_pre_close] - 1.0) * 100.0
    pp_data[:,PPI_close_increase] = (pp_data[:,PPI_close] / pp_data[:,PPI_pre_close] - 1.0) * 100.0
    pp_data[:,PPI_high_increase] = (pp_data[:,PPI_high] / pp_data[:,PPI_pre_close] - 1.0) * 100.0
    pp_data[:,PPI_low_increase] = (pp_data[:,PPI_low] / pp_data[:,PPI_pre_close] - 1.0) * 100.0

    # region 5
    r5_len = data_len - 4
    pp_data[:r5_len, PPI_open_5] = pp_data[4:4+r5_len, PPI_open]
    pp_data[:r5_len, PPI_close_5] = pp_data[:r5_len, PPI_close]
    pp_data[:r5_len, PPI_high_5] = np.max(PPSpread(pp_data, PPI_high, 5), axis=1)
    pp_data[:r5_len, PPI_low_5] = np.min(PPSpread(pp_data, PPI_low, 5), axis=1)
    pp_data[:r5_len, PPI_vol_5] = np.sum(PPSpread(pp_data, PPI_vol, 5), axis=1)
    if use_daily_basic:
        pp_data[:r5_len, PPI_turnover_rate_f_5] = np.sum(PPSpread(pp_data, PPI_turnover_rate_f, 5), axis=1)
    
    # avg
    pp_data[:data_len-(5-1), PPI_close_5_avg] = np.mean(PPSpread(pp_data, PPI_close, 5), axis=1)
    pp_data[:data_len-(10-1), PPI_close_10_avg] = np.mean(PPSpread(pp_data, PPI_close, 10), axis=1)
    pp_data[:data_len-(30-1), PPI_close_30_avg] = np.mean(PPSpread(pp_data, PPI_close, 30), axis=1)
    pp_data[:data_len-(100-1), PPI_close_100_avg] = np.mean(PPSpread(pp_data, PPI_close, 100), axis=1)
    pp_data[:data_len-(5-1), PPI_vol_5_avg] = np.mean(PPSpread(pp_data, PPI_vol, 5), axis=1)
    pp_data[:data_len-(10-1), PPI_vol_10_avg] = np.mean(PPSpread(pp_data, PPI_vol, 10), axis=1)
    pp_data[:data_len-(30-1), PPI_vol_30_avg] = np.mean(PPSpread(pp_data, PPI_vol, 30), axis=1)
    pp_data[:data_len-(100-1), PPI_vol_100_avg] = np.mean(PPSpread(pp_data, PPI_vol, 100), axis=1)
    
    StockDataPreProcess_AddSuspendBorder(pp_data)
    return pp_data[:data_len - preprocess_ref_days]

def StockDataPreProcessv1(stock_data_df, adj_mode):
    src_basic_col_names_str = [
        'ts_code',
        'trade_date'
    ]
    src_basic_col_names_float = [
        'open', 
        'close', 
        'high', 
        'low', 
        'vol'
    ]
    src_daily_basic_col_names = [
        'turnover_rate_f'
    ]
    src_moneyflow_col_names = [
        'buy_sm_vol',
        'sell_sm_vol',
        'buy_md_vol',
        'sell_md_vol',
        'buy_lg_vol',
        'sell_lg_vol',
        'buy_elg_vol',
        'sell_elg_vol',
        'net_mf_vol'
    ]
    src_adj_factor_col_names = ['adj_factor']
    src_avg_col_names = [
        'close', 
        'vol'
    ]
    use_daily_basic = stock_data_df.columns.contains(src_daily_basic_col_names[0]);
    use_money_flow = stock_data_df.columns.contains(src_moneyflow_col_names[0]);
    use_adj_factor = stock_data_df.columns.contains(src_adj_factor_col_names[0]);

    src_all_col_names = src_basic_col_names_str + src_basic_col_names_float
    src_float_col_names = src_basic_col_names_float
    if use_daily_basic:
        src_all_col_names = src_all_col_names + src_daily_basic_col_names
        src_float_col_names = src_float_col_names + src_daily_basic_col_names
    if use_money_flow:
        src_all_col_names = src_all_col_names + src_moneyflow_col_names
        src_float_col_names = src_float_col_names + src_moneyflow_col_names
    if use_adj_factor:
        src_all_col_names = src_all_col_names + src_adj_factor_col_names
        src_float_col_names = src_float_col_names + src_adj_factor_col_names

    if len(stock_data_df) == 0:
        return stock_data_df
    src_df_2=stock_data_df[src_all_col_names].copy().reset_index(drop=True)

    if (adj_mode != '') and (not use_adj_factor):
        print('StockDataPreProcess.Error: adj_mode=%s, use_adj_factor=%u' % (adj_mode, int(use_adj_factor)))
        return []
    if adj_mode == 'f':  # forward
        StockDataPreProcess_AdjForward(src_df_2)

    src_df_2['pre_close']=0.0
    for day_loop in range(0, len(src_df_2) - 1): 
        src_df_2.loc[day_loop,'pre_close'] = src_df_2.loc[day_loop + 1,'close']
    src_df_2.loc[len(src_df_2) - 1,'pre_close'] = src_df_2.loc[len(src_df_2) - 1,'close']

    src_df_2['open_increase']=0.0
    src_df_2['close_increase']=0.0
    src_df_2['high_increase']=0.0
    src_df_2['low_increase']=0.0
    src_df_2['open_5']=0.0
    src_df_2['close_5']=0.0
    src_df_2['high_5']=0.0
    src_df_2['low_5']=0.0
    if use_daily_basic:
        src_df_2['turnover_rate_f_5']=0.0
    src_df_2['vol_5']=0.0
    src_df=src_df_2.copy()
    if len(src_df) < preprocess_ref_days:
        return src_df[0:0]

    for day_loop in range(0, len(src_df)):
        for col_name in src_float_col_names:
            if math.isnan(src_df.loc[day_loop, col_name]):
                print('StockDataPreProcess.Error1, %s[%d] is nan' %(col_name, day_loop))
                return src_df[0:0]  
        if src_df.loc[day_loop,'trade_date'] == '' \
                or src_df.loc[day_loop,'open'] == 0.0 \
                or src_df.loc[day_loop,'close'] == 0.0 \
                or src_df.loc[day_loop,'high'] == 0.0 \
                or src_df.loc[day_loop,'low'] == 0.0:
            print('StockDataPreProcess.Error2, %s, %f, %f, %f, %f' %(src_df.loc[day_loop,'trade_date'], \
                                                                    src_df.loc[day_loop,'open'], \
                                                                    src_df.loc[day_loop,'close'], \
                                                                    src_df.loc[day_loop,'high'], \
                                                                    src_df.loc[day_loop,'low']))
            return src_df[0:0]     
    
    for col_name in src_avg_col_names:
        StockDataPreProcess_AddAvg(src_df, col_name, 5)
        StockDataPreProcess_AddAvg(src_df, col_name, 10)
        StockDataPreProcess_AddAvg(src_df, col_name, 30)
        StockDataPreProcess_AddAvg(src_df, col_name, 100)
        # StockDataPreProcess_AddAvg(src_df, col_name, 200)

    loop_count = 0
    for day_loop in reversed(range(0, len(src_df))):
        # open_5, close_5, high_5, low_5
        if loop_count >= 5:
            src_df.loc[day_loop, 'open_5'] = src_df.loc[day_loop + 4, 'open']
            src_df.loc[day_loop, 'close_5'] = src_df.loc[day_loop, 'close']
            high_5 = 0
            low_5 = 100000.0
            vol_5_sum = 0.0
            for iloop in range(0, 5):
                temp_index = day_loop + iloop
                if high_5 < src_df.loc[temp_index, 'high']:
                    high_5 = src_df.loc[temp_index, 'high']
                if low_5 > src_df.loc[temp_index, 'low']:
                    low_5 = src_df.loc[temp_index, 'low']
                vol_5_sum += src_df.loc[temp_index, 'vol']
            src_df.loc[day_loop, 'high_5'] = high_5
            src_df.loc[day_loop, 'low_5'] = low_5
            # if use_turnover_rate_f:
            #     src_df.loc[day_loop, 'turnover_rate_f_5'] = trf_5_sum
            src_df.loc[day_loop, 'vol_5'] = vol_5_sum

        loop_count += 1
            
    temp_pre_close = 0.0
    for day_loop in range(0, len(src_df)):
        temp_pre_close = src_df.loc[day_loop,'pre_close']
        if temp_pre_close == 0.0:
            print('Error: pre_close == %f, trade_date: %s' % (src_df.loc[day_loop,'pre_close'], src_df.loc[day_loop,'trade_date']))
            return src_df[0:0]
        src_df.loc[day_loop,'open_increase'] = ((src_df.loc[day_loop,'open'] / temp_pre_close) - 1.0) * 100.0
        src_df.loc[day_loop,'close_increase'] = ((src_df.loc[day_loop,'close'] / temp_pre_close) - 1.0) * 100.0
        src_df.loc[day_loop,'high_increase'] = ((src_df.loc[day_loop,'high'] / temp_pre_close) - 1.0) * 100.0
        src_df.loc[day_loop,'low_increase'] = ((src_df.loc[day_loop,'low'] / temp_pre_close) - 1.0) * 100.0
    StockDataPreProcess_AddSuspendBorder(src_df)
    # if use_adj_factor:
    #     StockDataPreProcess_AddAdjFlag(src_df)
    return src_df[:len(src_df)-preprocess_ref_days]



if __name__ == "__main__":

    print('main')
