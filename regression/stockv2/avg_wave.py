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
                 continue_down_num, 
                 cut_loss_ratio,
                 dataset_sample_num = 5):
        class_name = 'avg_wave'
        self.avg_cycle = avg_cycle
        self.mode = mode
        self.continue_up_num = continue_up_num
        self.continue_down_num = continue_down_num
        self.data_source = o_data_source
        self.feature = o_feature
        self.dataset_size = 0
        self.dataset_len = 0
        app_setting_name = '%u_%u_%u' % (avg_cycle, mode, continue_up_num)
        super(AvgWave, self).__init__(o_data_source, 
                                      o_feature, 
                                      class_name, 
                                      app_setting_name,
                                      -100.0,
                                      dataset_sample_num,
                                      cut_loss_ratio)
        self.InitDSWWsUpNumDataset()
        self.code_num = len(self.data_source.code_list)

    def TradePP(self, pp_data):
        data_len = len(pp_data)
        if data_len == 0:
            return
        self.wave_data = np.zeros((data_len), dtype=np.int)
        continue_up_count = 0
        continue_down_count = 0
        if self.mode == MODE_GRAD:
            for day_loop in reversed(range(0, data_len-1)):
                if pp_data[day_loop][self.avg_cycle] > pp_data[day_loop + 1][self.avg_cycle]:
                    continue_up_count += 1
                else:
                    continue_up_count = 0
                    continue_down_count += 1
                if continue_up_count > 0 and \
                    continue_up_count >= self.continue_up_num and \
                    continue_down_count >= self.continue_down_num:
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

        # if self.wave_data[day_index] == WS_UP:
        #     if self.GetDSWWsUpNum(pp_data[day_index][PPI_trade_date]) > self.code_num * 0.25:
        #         return TS_ON
        #     else:
        #         return TS_OFF
        # else:
        #     return TS_OFF
        if self.wave_data[day_index] == WS_UP:
            return TS_ON
        else:
            return TS_OFF

def main(argv):
    del argv
    code_filter = '\
000001.SZ,\
000002.SZ,\
000063.SZ,\
000538.SZ,\
000541.SZ,\
000550.SZ,\
000560.SZ,\
000561.SZ,\
000584.SZ,\
000625.SZ,\
000650.SZ,\
000651.SZ,\
000721.SZ,\
000800.SZ,\
000802.SZ,\
000858.SZ,\
000898.SZ,\
000913.SZ,\
000927.SZ,\
000932.SZ,\
000937.SZ,\
000938.SZ,\
000951.SZ,\
000959.SZ,\
001696.SZ,\
600000.SH,\
600006.SH,\
600085.SH,\
600104.SH,\
600109.SH,\
600115.SH,\
600137.SH,\
600177.SH,\
600198.SH,\
600199.SH,\
600600.SH,\
600601.SH,\
600609.SH,\
600612.SH,\
600623.SH,\
600624.SH,\
600664.SH,\
600679.SH,\
600702.SH,\
600718.SH,\
600809.SH'
    o_data_source = tushare_data.DataSource(20000101, '', code_filter, 1, 20130101, 20200106, False, False, True)
    o_feature = feature.Feature(30, feature.FUT_D5_NORM, 1, False, False)
    # o_feature = feature.Feature(30, feature.FUT_D5_NORM, 1, False, False)
    # o_feature = feature.Feature(30, feature.FUT_5REGION5_NORM, 5, False, False)
    # o_feature = feature.Feature(30, feature.FUT_2AVG5_NORM, 5, False, False)
    o_avg_wave = AvgWave(o_data_source, o_feature, PPI_close_30_avg, MODE_GRAD, 0, 0, 0.1)
    split_date = 20180101
    o_dl_model = dl_model.DLModel('%s_%u' % (o_avg_wave.setting_name, split_date), 
                                  o_feature.feature_unit_num, 
                                  o_feature.feature_unit_size,
                                  32, 10240, 0.004, 'mean_absolute_tp0_max_ratio_error')
    if FLAGS.mode == 'data':
        o_data_source.DownloadData()
        o_data_source.UpdatePPData()
    elif FLAGS.mode == 'testall':
        o_avg_wave.TradeTestAll(True, FLAGS.show)
    elif FLAGS.mode == 'test':
        o_data_source.DownloadStockData(FLAGS.c)
        o_data_source.UpdateStockPPData(FLAGS.c)
        start_time = time.time()
        o_avg_wave.TradeTestStock(FLAGS.c, FLAGS.show)
        print(time.time() - start_time)
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
    
