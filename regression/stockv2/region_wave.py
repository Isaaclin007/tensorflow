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

class RegionWave(trade_base.TradeBase):
    def __init__(self,
                 o_data_source,
                 o_feature,
                 region_days,
                 continue_up_num = 1,
                 cut_loss_ratio = 0.1,
                 dataset_sample_num = 5):
        class_name = 'region_wave'
        self.region_days = region_days
        self.continue_up_num = continue_up_num
        self.data_source = o_data_source
        self.feature = o_feature
        self.dataset_size = 0
        self.dataset_len = 0
        app_setting_name = '%u' % (region_days)
        super(RegionWave, self).__init__(o_data_source, 
                                      o_feature, 
                                      class_name, 
                                      app_setting_name,
                                      -100.0,
                                      dataset_sample_num,
                                      cut_loss_ratio)

    def TradePP(self, pp_data):
        data_len = len(pp_data)
        if data_len <= self.region_days:
            return
        self.wave_data = np.zeros((data_len), dtype=np.int)
        continue_up_count = 0
        for day_loop in reversed(range(0, data_len-self.region_days)):
            region_max = np.max(pp_data[day_loop+1:day_loop+1+self.region_days, PPI_close])
            region_min = np.min(pp_data[day_loop+1:day_loop+1+self.region_days, PPI_close])
            if pp_data[day_loop][PPI_close] > region_max:
                continue_up_count += 1
            else:
                continue_up_count = 0
            if continue_up_count > 0 and continue_up_count >= self.continue_up_num:
                self.wave_data[day_loop] = WS_UP
            elif pp_data[day_loop][PPI_close] < region_min:
                self.wave_data[day_loop] = WS_DOWN
            else:
                self.wave_data[day_loop] = WS_NONE
            # print('%-12u%u' % (pp_data[day_loop][PPI_trade_date], self.wave_data[day_loop]))
        
    def TradeNextStatus(self, pp_data, day_index):
        if self.wave_data[day_index] == WS_UP:
            return TS_ON
        elif self.wave_data[day_index] == WS_DOWN:
            return TS_OFF
        elif self.wave_data[day_index] == WS_NONE:
            return TS_NONE

def main(argv):
    del argv

    o_data_source = tushare_data.DataSource(20000101, '', '', 1, 20000101, 20200106, False, False, True)
    o_feature = feature.Feature(30, feature.FUT_D5_NORM, 1, False, False)
    o_avg_wave = RegionWave(o_data_source, o_feature, 100)
    split_date = 20180101
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
        o_avg_wave.TradeTestStock(FLAGS.c, FLAGS.show)
    elif FLAGS.mode == 'train':
        tf, tl, vf, vl, td = o_avg_wave.GetDataset(split_date)
        tl = tl * 100.0
        vl  = vl * 100.0
        o_dl_model.Train(tf, tl, vf, vl, FLAGS.epoch)
    elif FLAGS.mode == 'rtest':
        tf, tl, tf, tl, ta = o_avg_wave.GetDataset(split_date)
        o_dl_model.LoadModel(FLAGS.epoch)
        o_avg_wave.RTest(o_dl_model, tf, ta, False)
    elif FLAGS.mode == 'dsw':
        dataset = o_avg_wave.ShowDSW3DDataset()
    elif FLAGS.mode == 'show':
        dataset = o_avg_wave.ShowTradePP(FLAGS.c)

    exit()

if __name__ == "__main__":
    flags.DEFINE_string('mode', 'test', 'test | testall | train')
    flags.DEFINE_string('c', '000001.SZ', 'ts code')
    flags.DEFINE_integer('epoch', 1000, 'train or rtest epoch')
    flags.DEFINE_boolean('show', False, 'show trade record')
    app.run(main)
    