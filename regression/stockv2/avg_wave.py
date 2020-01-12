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

reload(sys)
sys.setdefaultencoding('utf-8')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

FLAGS = flags.FLAGS

MODE_GRAD = 0
MODE_GRAD_GRAD = 1
mode = MODE_GRAD
continue_up_num = 1

class AvgWave(trade_base.TradeBase):
    def __init__(self,
                 o_data_source,
                 o_feature,
                 avg_cycle,
                 mode,
                 continue_up_num, 
                 cut_loss_ratio):
        class_name = 'avg_wave'
        self.avg_cycle = avg_cycle
        self.mode = mode
        self.continue_up_num = continue_up_num
        self.cut_loss_ratio = cut_loss_ratio
        self.data_source = o_data_source
        self.feature = o_feature
        self.dataset_size = 0
        self.dataset_len = 0
        app_setting_name = '%u_%u_%u_%.4f' % (avg_cycle, mode, continue_up_num, cut_loss_ratio)
        self.index_increase     = self.feature.feature_size
        self.index_ts_code      = self.feature.feature_size + 1
        self.index_pre_on_date  = self.feature.feature_size + 2
        self.index_on_date      = self.feature.feature_size + 3
        self.index_pre_off_date = self.feature.feature_size + 4
        self.index_off_date     = self.feature.feature_size + 5
        self.index_holding_days = self.feature.feature_size + 6
        super(AvgWave, self).__init__(o_data_source, 
                                      o_feature, 
                                      class_name, 
                                      app_setting_name,
                                      0.0)

    def TradePP(self, pp_data):
        data_len = len(pp_data)
        if data_len == 0:
            return
        self.wave_data = np.zeros((data_len), dtype=np.int)
        continue_up_count = 0
        if self.mode == MODE_GRAD:
            for day_loop in reversed(range(0, data_len-1)):
                if pp_data[day_loop][self.avg_cycle] > pp_data[day_loop + 1][self.avg_cycle]:
                    continue_up_count += 1
                else:
                    continue_up_count = 0
                if continue_up_count >= self.continue_up_num:
                    self.wave_data[day_loop] = WS_UP
                else:
                    self.wave_data[day_loop] = WS_DOWN
        # elif self.mode == MODE_GRAD_GRAD:
        #     grad_data = np_common.Grad(pp_data[avg_name].values)
        #     grad_data = np_common.Grad(grad_data)
        #     for day_loop in reversed(range(0, data_len-1)):
        #         if grad_data[day_loop] > 0:
        #             continue_up_count += 1
        #         else:
        #             continue_up_count = 0
        #         if continue_up_count >= self.continue_up_num:
        #             pp_data.loc[day_loop, wave_data_name] = WS_UP
        #         else:
        #             pp_data.loc[day_loop, wave_data_name] = WS_DOWN
        
    def TradeNextStatus(self, pp_data, day_index):
        if self.wave_data[day_index] == WS_UP:
            return TS_ON
        else:
            return TS_OFF

def main(argv):
    del argv

    o_data_source = tushare_data.DataSource(20000101, '', '', 1, 20000101, 20200106, False, False, True)
    # o_feature = feature.Feature(30, feature.FUT_D5_NORM, 1, False, False)
    # o_feature = feature.Feature(30, feature.FUT_D5_NORM, 1, False, False)
    o_feature = feature.Feature(30, feature.FUT_5REGION5_NORM, 5, False, False)
    o_avg_wave = AvgWave(o_data_source, o_feature, PPI_close_30_avg, MODE_GRAD, 1, 0.1)
    split_date = 20170101
    o_dl_model = dl_model.DLModel('%s_%u' % (o_avg_wave.setting_name, split_date), 
                                  o_feature.feature_unit_num, 
                                  o_feature.feature_unit_size,
                                  32, 10240, 0.004, 'mean_absolute_tp0_max_ratio_error')
    if FLAGS.mode == 'data':
        o_data_source.DownloadData()
        o_data_source.UpdatePPData()
    elif FLAGS.mode == 'testall':
        o_avg_wave.TradeTestAll()
    elif FLAGS.mode == 'test':
        start_time = time.time()
        o_avg_wave.TradeTestStock(FLAGS.c)
        print(time.time() - start_time)
    elif FLAGS.mode == 'train':
        tf, tl, vf, vl, td = o_avg_wave.GetDataset(split_date)
        tl = tl * 100.0
        vl  = vl * 100.0
        o_dl_model.Train(tf, tl, vf, vl, 100)
    elif FLAGS.mode == 'rtest':
        tf, tl, tf, tl, ta = o_avg_wave.GetDataset(split_date)
        o_dl_model.LoadModel()
        o_avg_wave.RTest(o_dl_model, tf, ta, False)
        
    exit()

if __name__ == "__main__":
    flags.DEFINE_string('mode', 'test', 'test | testall | train')
    flags.DEFINE_string('c', '000001.SZ', 'ts code')
    # flags.DEFINE_boolean('testall', False, 'test all stocks, save dataset')
    app.run(main)
    