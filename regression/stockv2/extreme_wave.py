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

# min_wave_width_left = 2
# min_wave_width_right = 2
# trade_off_threshold = 0
# up_100avg_condition = True
# up_200avg_condition = False
# wave_index = 'close'
# save_original_dataset = False

EXTREME_NONE = 0
EXTREME_PEAK = 1
EXTREME_VALLEY = 2
EXTREME_STR = ['NONE', 'PEAK', 'VALLEY']

STATUS_NONE = 0
STATUS_UP = 1
STATUS_DOWN = 2
WAVE_STATUS_STR = ['NONE', 'UP', 'DOWN']

# wave_test_dataset_sample_num = 5

class ExtremeWave(trade_base.TradeBase):
    def __init__(self,
                 o_data_source,
                 o_feature, 
                 min_wave_width_left = 2,
                 min_wave_width_right = 2,
                 up_100avg_condition = True,
                 trade_off_threshold = 0,
                 cut_loss_ratio = 0.1,
                 dataset_sample_num = 5):
        class_name = 'extreme_wave'
        # inherit class setting
        self.min_wave_width_left = min_wave_width_left
        self.min_wave_width_right = min_wave_width_right
        self.up_100avg_condition = up_100avg_condition
        self.trade_off_threshold = trade_off_threshold
        app_setting_name = '%u_%u_%u_%u_%u' % (min_wave_width_left, 
                                         min_wave_width_right, 
                                         int(up_100avg_condition),
                                         trade_off_threshold, 
                                         dataset_sample_num)
        super(ExtremeWave, self).__init__(o_data_source, 
                                      o_feature, 
                                      class_name, 
                                      app_setting_name,
                                      0.0,
                                      dataset_sample_num,
                                      cut_loss_ratio)
    
    # 返回 extreme_flag, extreme_value
    def FitExtreme(self, pp_data, data_len, day_index):
        if (data_len - day_index) < (self.min_wave_width_left + self.min_wave_width_right + 1):
            return EXTREME_NONE, 0.0
        middle_index = day_index + self.min_wave_width_right
        middle_value = pp_data[middle_index][PPI_close]
        is_peak = True
        is_valley = True
        for iloop in range(1, self.min_wave_width_left + 1):
            temp_value = pp_data[middle_index + iloop][PPI_close]
            if middle_value > temp_value:
                is_valley = False
            elif middle_value < temp_value:
                is_peak = False
        for iloop in range(1, self.min_wave_width_right + 1):
            temp_value = pp_data[middle_index - iloop][PPI_close]
            if middle_value >= temp_value:
                is_valley = False
            elif middle_value <= temp_value:
                is_peak = False
        # 计算 extreme_flag 当天峰谷标志
        if is_peak:
            extreme_flag = EXTREME_PEAK
        elif is_valley:
            extreme_flag = EXTREME_VALLEY
        else:
            extreme_flag = EXTREME_NONE
        return extreme_flag, middle_value

    def TradePP(self, pp_data):
        data_len = len(pp_data)
        if data_len == 0:
            return
        self.wave_data = np.zeros((data_len), dtype=np.int)
        self.trade_flag = np.zeros((data_len), dtype=np.int)
        last_peak = []
        last_valley = []
        trade_off_count = 0
        last_extreme = EXTREME_NONE
        current_status = STATUS_NONE
        last_flag = TS_NONE

        for day_loop in reversed(range(0, data_len)):
            day_index_reversed = data_len - day_loop - 1  # day_loop 倒数的 index
            close = pp_data[day_loop][PPI_close]
            close_10_avg = pp_data[day_loop][PPI_close_10_avg]
            close_100_avg = pp_data[day_loop][PPI_close_100_avg]
            fit_flag, e_value = self.FitExtreme(pp_data, data_len, day_loop)

            # set self.trade_flag
            if fit_flag == EXTREME_PEAK:
                if len(last_peak) > 0 and e_value <= last_peak[0]:
                    self.trade_flag[day_loop] = TS_OFF
            elif fit_flag == EXTREME_VALLEY:
                if len(last_valley) > 0 and e_value <= last_valley[0]:
                    self.trade_flag[day_loop] = TS_OFF

            if len(last_valley) > 0 and close < last_valley[0]:
                self.trade_flag[day_loop] = TS_OFF
            if (len(last_peak) > 0 and close > last_peak[0]) and \
               ((not self.up_100avg_condition) or (close_10_avg > close_100_avg)) and \
               (trade_off_count >= self.trade_off_threshold):
                self.trade_flag[day_loop] = TS_ON

            # set last_peak,last_valley,last_extreme,self.wave_data
            if fit_flag == EXTREME_PEAK:
                last_peak.insert(0, e_value)
                last_extreme = EXTREME_PEAK
                current_status = WS_DOWN
            elif fit_flag == EXTREME_VALLEY:
                last_valley.insert(0, e_value)
                last_extreme = EXTREME_VALLEY
                current_status = WS_UP
            self.wave_data[day_loop] = current_status

            # set trade_off_count
            if last_flag == TS_ON:
                trade_off_count = 0
            else:
                trade_off_count += 1
            if self.trade_flag[day_loop] != TS_NONE:
                last_flag = self.trade_flag[day_loop]



        
    def TradeNextStatus(self, pp_data, day_index):
        trade_flag = self.trade_flag[day_index]
        return trade_flag

def main(argv):
    del argv

    o_data_source = tushare_data.DataSource(20000101, '', '', 1, 20120101, 20200106, False, False, True)
    o_feature = feature.Feature(30, feature.FUT_D5_NORM, 1, False, False)
    # o_feature = feature.Feature(30, feature.FUT_D5_NORM, 1, False, False)
    # o_feature = feature.Feature(30, feature.FUT_5REGION5_NORM, 5, False, False)
    # o_feature = feature.Feature(30, feature.FUT_2AVG5_NORM, 5, False, False)
    o_wave = ExtremeWave(o_data_source, o_feature, 2, 2, False, 0, 0.1, 5)
    split_date = 20180101
    o_dl_model = dl_model.DLModel('%s_%u' % (o_wave.setting_name, split_date), 
                                  o_feature.feature_unit_num, 
                                  o_feature.feature_unit_size,
                                  32, 10240, 0.004, 'mean_absolute_tp0_max_ratio_error')
    if FLAGS.mode == 'data':
        o_data_source.DownloadData()
        o_data_source.UpdatePPData()
    elif FLAGS.mode == 'testall':
        o_wave.TradeTestAll()
    elif FLAGS.mode == 'test':
        o_wave.TradeTestStock(FLAGS.c, FLAGS.show)
    elif FLAGS.mode == 'show':
        o_wave.ShowTradePP(FLAGS.c)
    elif FLAGS.mode == 'train':
        tf, tl, vf, vl, td = o_wave.GetDataset(split_date)
        tl = tl * 100.0
        vl  = vl * 100.0
        o_dl_model.Train(tf, tl, vf, vl, FLAGS.epoch)
    elif FLAGS.mode == 'rtest':
        tf, tl, tf, tl, ta = o_wave.GetDataset(split_date)
        o_dl_model.LoadModel(FLAGS.epoch)
        o_wave.RTest(o_dl_model, tf, ta, False)
        
    exit()

if __name__ == "__main__":
    flags.DEFINE_string('mode', 'test', 'test | testall | train')
    flags.DEFINE_string('c', '000001.SZ', 'ts code')
    flags.DEFINE_integer('epoch', 100, 'train or rtest epoch')
    flags.DEFINE_boolean('show', False, 'show trade record')
    app.run(main)
    