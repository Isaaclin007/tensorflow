# -*- coding:UTF-8 -*-

import tensorflow as tf
from tensorflow import keras
import tushare as ts
import numpy as np
import pandas as pd
import os
import time
import sys
from absl import app
from absl import flags
import tushare_data
import random
sys.path.append("..")
from common.base_common import *
from common import np_common

reload(sys)
sys.setdefaultencoding('utf-8')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

FLAGS = flags.FLAGS

MODE_GRAD = 0
MODE_GRAD_GRAD = 1
mode = MODE_GRAD
continue_up_num = 1

def AppendWaveData(pp_data, avg_cycle=30):
    data_len = len(pp_data)
    if data_len == 0:
        return
    pp_data['avg_wave_status'] = WS_NONE
    
    avg_name = 'close_%u_avg' % avg_cycle
    continue_up_count = 0
    if mode == MODE_GRAD:
        grad_data = np_common.Grad(pp_data[avg_name].values)
        for day_loop in reversed(range(0, data_len-1)):
            if grad_data[day_loop] > 0:
                continue_up_count += 1
            else:
                continue_up_count = 0
            if continue_up_count >= continue_up_num:
                pp_data.loc[day_loop, 'avg_wave_status'] = WS_UP
            else:
                pp_data.loc[day_loop, 'avg_wave_status'] = WS_DOWN
    elif mode == MODE_GRAD_GRAD:
        grad_data = np_common.Grad(pp_data[avg_name].values)
        grad_data = np_common.Grad(grad_data)
        for day_loop in reversed(range(0, data_len-1)):
            if grad_data[day_loop] > 0:
                continue_up_count += 1
            else:
                continue_up_count = 0
            if continue_up_count >= continue_up_num:
                pp_data.loc[day_loop, 'avg_wave_status'] = WS_UP
            else:
                pp_data.loc[day_loop, 'avg_wave_status'] = WS_DOWN
        

def TradeTest(pp_data, \
              avg_cycle, \
              cut_loss_ratio, \
              print_trade_flag):
    data_len = len(pp_data)
    if data_len == 0:
        return 0.0
    AppendWaveData(pp_data, avg_cycle)
    ts_code = pp_data.loc[0, 'ts_code']

    # global status
    trade_count = 0
    sum_increase = 0.0
    capital_ratio = 1.0
    sum_holding_days = 0
    trade_status = TS_OFF

    # trade status
    pre_on_date = INVALID_DATE
    on_day_index = 0
    on_price = 0
    on_date = INVALID_DATE
    pre_off_date = INVALID_DATE
    off_price = 0
    off_date = INVALID_DATE
    increase = 0.0
    holding_days = 0

    for day_index in reversed(range(0, data_len)):
        if pp_data.loc[day_index, 'avg_wave_status'] == WS_UP:
            next_status = TS_ON
        else:
            next_status = TS_OFF

        if trade_status == TS_OFF:
            if next_status == TS_ON:
                trade_status = TS_PRE_ON
                pre_on_date = pp_data.loc[day_index,'trade_date']
        elif trade_status == TS_PRE_ON:
            trade_status = TS_ON
            on_day_index = day_index
            on_price = pp_data.loc[day_index, 'open']
            on_date = pp_data.loc[day_index,'trade_date']
        elif trade_status == TS_ON:
            if next_status == TS_OFF:
                trade_status = TS_PRE_OFF
                pre_off_date = pp_data.loc[day_index,'trade_date']
            
        elif trade_status == TS_PRE_OFF:
            trade_status = TS_OFF
            off_price = pp_data.loc[day_index, 'open']
            off_date = pp_data.loc[day_index,'trade_date']
            increase = off_price / on_price - 1.0
            holding_days = on_day_index - day_index
            sum_increase += increase
            capital_ratio *= (1.0 + increase)
            sum_holding_days += holding_days
            PrintTrade(trade_count, ts_code, pre_on_date, on_date, pre_off_date, off_date, increase, holding_days)
            trade_count += 1

            # trade status
            pre_on_date = INVALID_DATE
            on_day_index = 0
            on_price = 0
            on_date = INVALID_DATE
            pre_off_date = INVALID_DATE
            off_day_index = 0
            off_price = 0
            off_date = INVALID_DATE
            increase = 0.0
            holding_days = 0
    if trade_status == TS_PRE_ON:
        PrintTrade(trade_count, ts_code, pre_on_date, INVALID_DATE, INVALID_DATE, INVALID_DATE, '--', '--')
    elif trade_status == TS_ON:
        day_index = 0
        off_price = pp_data.loc[day_index, 'open']
        increase = off_price / on_price - 1.0
        holding_days = on_day_index - day_index
        PrintTrade(trade_count, ts_code, pre_on_date, on_date, INVALID_DATE, INVALID_DATE, increase, holding_days)
    elif trade_status == TS_PRE_OFF:
        day_index = 0
        off_price = pp_data.loc[day_index, 'open']
        increase = off_price / on_price - 1.0
        holding_days = on_day_index - day_index
        PrintTrade(trade_count, ts_code, pre_on_date, on_date, pre_off_date, INVALID_DATE, increase, holding_days)

    PrintTrade('sum', '--', '--', '--', '--', '--', capital_ratio, sum_holding_days)
    return sum_increase


def main(argv):
    del argv

    stock_pp_file_name = tushare_data.FileNameStockPreprocessedData(FLAGS.c)
    if os.path.exists(stock_pp_file_name):
        pp_data = pd.read_csv(stock_pp_file_name)
        TradeTest(pp_data, FLAGS.avg, 0.1, True)
    else:
        print("File not exist: %s" % stock_pp_file_name)
        

    exit()
if __name__ == "__main__":
    flags.DEFINE_string('c', '--', 'ts code')
    flags.DEFINE_integer('avg', 30, 'avg cycle')
    app.run(main)
    