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
import random
sys.path.append("..")
from common import base_common
from common import np_common
from common.const_def import *
import tushare_data
import feature
import dl_model
import trade_base
import preprocess
from trade_state import *

reload(sys)
sys.setdefaultencoding('utf-8')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

FLAGS = flags.FLAGS

# leverage: 股指期货杠杆
class Breakup(trade_base.TradeBase):
    def __init__(self,
                 o_data_source,
                 o_feature,
                 avg_cycle,
                 continue_up_num, 
                 continue_down_num,
                 leverage = 1.0,
                 cut_loss_ratio = 0.1,
                 dataset_sample_num = 5):
        class_name = 'breakup'
        self.avg_cycle = avg_cycle
        self.continue_up_num = continue_up_num
        self.continue_down_num = continue_down_num
        self.leverage = leverage
        self.data_source = o_data_source
        self.feature = o_feature
        self.dataset_size = 0
        self.dataset_len = 0
        app_setting_name = '%u_%u_%u' % (avg_cycle,
                                            continue_up_num, 
                                            continue_down_num)
        super(Breakup, self).__init__(o_data_source, 
                                      o_feature, 
                                      class_name, 
                                      app_setting_name,
                                      -100.0,
                                      dataset_sample_num,
                                      cut_loss_ratio)

    def funcname(parameter_list):
        pass

    # 待优化：
    #     1：continue down 的计算，如果没有大幅上涨计时breakup也不重置计数器
    #     2：trade off 条件，如果increase是正的，breakdown不trade off
    def TradePP_(self, pp_data):
        data_len = len(pp_data)
        if data_len == 0:
            return
        self.wave_data = np.zeros((data_len), dtype=np.int)
        continue_up_count = 0
        continue_down_count = 0
        last_status = WS_NONE
        current_status = WS_NONE
        current_ts = TS_NONE
        min_down_value = 1000000.0
        max_up_value = 0.0
        for day_loop in reversed(range(0, data_len)):
            close = pp_data[day_loop][PPI_close]
            avg = pp_data[day_loop][self.avg_cycle]
            # current_status
            if close > avg:
                current_status = WS_UP
            else:
                current_status = WS_DOWN
            
            # reset count
            if last_status == WS_UP and current_status == WS_DOWN:
                if base_common.IncPct(max_up_value, min_down_value) > 50:
                    continue_down_count = 0
                    min_down_value = 1000000.0
                continue_up_count = 0
                max_up_value = 0.0
            last_status = current_status

            # count
            if current_status == WS_UP:
                continue_up_count += 1
                if max_up_value < close:
                    max_up_value = close
            if current_status == WS_DOWN:
                continue_down_count += 1
                if min_down_value > close:
                    min_down_value = close
            
            # wave data
            if current_ts != TS_ON:
                if continue_up_count > 0 and \
                   continue_up_count >= self.continue_up_num and \
                   continue_down_count >= self.continue_down_num:
                    self.wave_data[day_loop] = TS_ON
                    current_ts = TS_ON
            if current_ts != TS_OFF:
                if current_status == WS_DOWN:
                    self.wave_data[day_loop] = TS_OFF
                    current_ts = TS_OFF
                if max_up_value > 0 and base_common.IncPct(close, max_up_value) < -20.0:
                    self.wave_data[day_loop] = TS_OFF
                    current_ts = TS_OFF
                    continue_down_count = 0
                    min_down_value = 1000000.0

            # print('%.0f, %.1f, %.1f, %.1f, %u' % (pp_data[day_loop][PPI_trade_date], 
            #                                   max_up_value, 
            #                                   close, 
            #                                   base_common.IncPct(close, max_up_value),
            #                                   self.wave_data[day_loop]))
        

    def TradePP(self, pp_data):
        data_len = len(pp_data)
        if data_len == 0:
            return
        self.wave_data = np.zeros((data_len), dtype=np.int)
        for day_loop in reversed(range(0, data_len)):
            close = pp_data[day_loop][PPI_close]
            avg = pp_data[day_loop][self.avg_cycle]
            if close > avg:
                self.wave_data[day_loop] = TS_ON
            else:
                self.wave_data[day_loop] = TS_OFF
            


    def TradeNextStatus(self, pp_data, day_index):
        return self.wave_data[day_index]

    def Test(self, ts_code):
        pp_data = self.data_source.LoadStockPPData(ts_code, True)
        preprocess.PPRegionCompute(pp_data, PPI_close, PPI_close_5_avg, 400, len(pp_data), np.mean)
        # pp_data = pp_data[pp_data[:,PPI_trade_date] >= 20100101]
        pp_data = pp_data[:-400]
        data_len = len(pp_data)
        if data_len == 0:
            return
        p = []
        lp = TradePos()
        sp = TradePos(True)
        trade_state = TradeState()
        p.append(lp)
        p.append(sp)
        current_status = WS_NONE
        sum_inc = 0.0
        capi_ratio = 1.0
        trade_count = 0
        continue_up_cnt = 0
        continue_down_cnt = 0
        for day_index in reversed(range(0, data_len)):
            open_price = pp_data[day_index][PPI_open]
            close_price = pp_data[day_index][PPI_close]
            close_500_avg = pp_data[day_index][PPI_close_5_avg]
            avg = pp_data[day_index][self.avg_cycle]
            temp_date = pp_data[day_index][PPI_trade_date]
            temp_status = WS_NONE
            if close_price > avg:
                continue_up_cnt += 1
                continue_down_cnt = 0
                if continue_up_cnt > self.continue_up_num:
                    temp_status = WS_UP
            else:
                continue_down_cnt += 1
                continue_up_cnt = 0
                if continue_down_cnt > self.continue_down_num:
                    temp_status = WS_DOWN

            for iloop in range(len(p)):
                if p[iloop].status == TS_PRE_ON:
                    p[iloop].status = TS_ON
                    p[iloop].on_date = temp_date
                    p[iloop].on_price = open_price
                elif p[iloop].status == TS_PRE_OFF:
                    p[iloop].status = TS_OFF
                    p[iloop].off_date = temp_date
                    p[iloop].off_price = open_price
                    p[iloop].UpdateInc()
                    p[iloop].inc *= self.leverage
                    sum_inc += p[iloop].inc
                    capi_ratio *= (1.0 + p[iloop].inc)
                    trade_count += 1
                    p[iloop].Print(trade_count, sum_inc, capi_ratio)
                    trade_state.AppendTrade(p[iloop].inc, capi_ratio, p[iloop].holding_days)
                    p[iloop].Reset()
                if p[iloop].status == TS_ON:
                    p[iloop].holding_days += 1
            # 趋势转换时，UP->DOWN 或 DOWN->UP，结束当前交易，开始新的交易
            if current_status != temp_status:
                current_status = temp_status
                # 非下跌，空头平仓
                if temp_status != WS_DOWN:
                    if sp.status != TS_OFF:
                        sp.status = TS_PRE_OFF
                        sp.pre_off_date = temp_date
                # 非上涨，多头平仓
                if temp_status != WS_UP:
                    if lp.status != TS_OFF:
                        lp.status = TS_PRE_OFF
                        lp.pre_off_date = temp_date
                # 上涨，多头开仓
                if temp_status == WS_UP and close_price < close_500_avg:
                # if temp_status == WS_UP:
                    if lp.status == TS_OFF:
                        lp.status = TS_PRE_ON
                        lp.pre_on_date = temp_date
                # 下跌，空头开仓
                if temp_status == WS_DOWN and close_price > close_500_avg:
                # if temp_status == WS_DOWN:
                    if sp.status == TS_OFF:
                        sp.status = TS_PRE_ON
                        sp.pre_on_date = temp_date
        for iloop in range(len(p)):
            if p[iloop].status != TS_OFF:
                p[iloop].Print(trade_count, sum_inc, capi_ratio)
        trade_state.Print()

def main(argv):
    del argv

    o_data_source = tushare_data.DataSource(20000101, '', '', 1, 20000101, 20200403, False, False, True)
    o_feature = feature.Feature(30, feature.FUT_D5_NORM_PCT, 1, False, False)
    # o_feature = feature.Feature(30, feature.FUT_D5_NORM, 1, False, False)
    # o_feature = feature.Feature(30, feature.FUT_5REGION5_NORM, 5, False, False)
    # o_feature = feature.Feature(30, feature.FUT_2AVG5_NORM, 5, False, False)
    o_trade = Breakup(o_data_source, o_feature, PPI_close_100_avg, 10, 10, 3.0, 0.1)
    split_date = 20180101
    o_dl_model = dl_model.DLModel('%s_%u' % (o_trade.setting_name, split_date), 
                                  o_feature.feature_unit_num, 
                                  o_feature.feature_unit_size,
                                  32, 10240, 0.004, 'mean_absolute_tp0_max_ratio_error')
    if FLAGS.mode == 'data':
        o_data_source.DownloadData()
        o_data_source.UpdatePPData()
    elif FLAGS.mode == 'testall':
        o_trade.TradeTestAll(True, FLAGS.show)
    elif FLAGS.mode == 'test':
        o_data_source.DownloadStockData(FLAGS.c)
        o_data_source.UpdateStockPPData(FLAGS.c)
        start_time = time.time()
        # o_trade.TradeTestStock(FLAGS.c, FLAGS.show)
        o_trade.Test(FLAGS.c)
    elif FLAGS.mode == 'train':
        tf, tl, vf, vl, td = o_trade.GetDataset(split_date)
        tl = tl * 100.0
        vl  = vl * 100.0
        o_dl_model.Train(tf, tl, vf, vl, FLAGS.epoch)
    elif FLAGS.mode == 'rtest':
        tf, tl, tf, tl, ta = o_trade.GetDataset(split_date)
        o_dl_model.LoadModel(FLAGS.epoch)
        o_trade.RTest(o_dl_model, tf, ta, False)
    elif FLAGS.mode == 'dsw':
        dataset = o_trade.ShowDSW3DDataset()
    elif FLAGS.mode == 'show':
        dataset = o_trade.ShowTradePP(FLAGS.c)

    exit()

if __name__ == "__main__":
    flags.DEFINE_string('mode', 'test', 'test | testall | train')
    flags.DEFINE_string('c', '000001.SZ', 'ts code')
    flags.DEFINE_integer('epoch', 100, 'train or rtest epoch')
    flags.DEFINE_boolean('show', False, 'show trade record')
    app.run(main)
    