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


class VolWave(trade_base.TradeBase):
    def __init__(self,
                 o_data_source,
                 o_feature,
                 cut_loss_ratio,
                 dataset_sample_num = 5):
        class_name = 'vol_wave'
        self.data_source = o_data_source
        self.feature = o_feature
        app_setting_name = ''
        super(VolWave, self).__init__(o_data_source, 
                                      o_feature, 
                                      class_name, 
                                      app_setting_name,
                                      -100.0,
                                      dataset_sample_num,
                                      cut_loss_ratio)
        self.code_num = len(self.data_source.code_list)

    def TradePP(self, pp_data):
        data_len = len(pp_data)
        if data_len == 0:
            return
        self.wave_data = np.zeros((data_len), dtype=np.int)
        for day_loop in reversed(range(0, data_len)):
            if pp_data[day_loop][PPI_vol] > (pp_data[day_loop][PPI_vol_10_avg] * 5):
                self.wave_data[day_loop] = WS_UP
            else:
                self.wave_data[day_loop] = WS_DOWN
        
    def TradeNextStatus(self, pp_data, day_index):
        if self.wave_data[day_index] == WS_UP:
            return TS_ON
        else:
            return TS_OFF

def main(argv):
    del argv
    o_data_source = tushare_data.DataSource(20000101, '', '', 1, 20100101, 20200306, False, False, True)
    o_feature = feature.Feature(7, feature.FUT_D5_NORM, 1, False, False)
    o_vol_wave = VolWave(o_data_source, o_feature, 0.1)
    # split_date = 20180101
    # o_dl_model = dl_model.DLModel('%s_%u' % (o_vol_wave.setting_name, split_date), 
    #                               o_feature.feature_unit_num, 
    #                               o_feature.feature_unit_size,
    #                               32, 10240, 0.004, 'mean_absolute_tp0_max_ratio_error')
    if FLAGS.mode == 'data':
        o_data_source.DownloadData()
        o_data_source.UpdatePPData()
    elif FLAGS.mode == 'testall':
        o_vol_wave.TradeTestAll(True, FLAGS.show)
    elif FLAGS.mode == 'test':
        o_data_source.DownloadStockData(FLAGS.c)
        o_data_source.UpdateStockPPData(FLAGS.c)
        start_time = time.time()
        o_vol_wave.TradeTestStock(FLAGS.c, FLAGS.show)
        print(time.time() - start_time)
    elif FLAGS.mode == 'train':
        tf, tl, vf, vl, td = o_vol_wave.GetDataset(split_date)
        tl = tl * 100.0
        vl  = vl * 100.0
        o_dl_model.Train(tf, tl, vf, vl, FLAGS.epoch)
    elif FLAGS.mode == 'rtest':
        tf, tl, tf, tl, ta = o_vol_wave.GetDataset(split_date)
        o_dl_model.LoadModel(FLAGS.epoch)
        o_vol_wave.RTest(o_dl_model, tf, ta, False)
    elif FLAGS.mode == 'dsw':
        dataset = o_vol_wave.ShowDSW3DDataset()
    elif FLAGS.mode == 'show':
        dataset = o_vol_wave.ShowTradePP(FLAGS.c)

    exit()

if __name__ == "__main__":
    flags.DEFINE_string('mode', 'test', 'test | testall | train')
    flags.DEFINE_string('c', '000001.SZ', 'ts code')
    flags.DEFINE_integer('epoch', 1000, 'train or rtest epoch')
    flags.DEFINE_boolean('show', False, 'show trade record')
    app.run(main)
    
