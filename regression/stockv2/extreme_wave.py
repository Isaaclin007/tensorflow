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
        self.cut_loss_ratio = cut_loss_ratio
        app_setting_name = '%u_%u_%u_%u_%f_%u' % (min_wave_width_left, 
                                         min_wave_width_right, 
                                         int(up_100avg_condition),
                                         trade_off_threshold, 
                                         cut_loss_ratio, 
                                         dataset_sample_num)
        super(ExtremeWave, self).__init__(o_data_source, 
                                      o_feature, 
                                      class_name, 
                                      app_setting_name,
                                      0.0,
                                      dataset_sample_num)
    def FillFirstWaveData(self, extreme_flag, start_day_index):
        if extreme_flag == EXTREME_PEAK:
            fill_status = STATUS_UP
        else:
            fill_status = STATUS_DOWN
        for day_loop in range(start_day_index, len(self.wave_extreme)):
            self.wave_extreme[day_loop] = EXTREME_NONE
            self.wave_status[day_loop] = fill_status

    def TradePP(self, pp_data):
        data_len = len(pp_data)
        if data_len == 0:
            return
        self.wave_extreme = np.zeros((data_len), dtype=np.int)
        self.wave_status = np.zeros((data_len), dtype=np.int)
        self.last_peak = -1.0
        self.last_valley = -1.0
        self.trade_off_count = 0
        last_extreme = EXTREME_NONE
        current_status = STATUS_NONE
        extreme_count = 0
        for day_loop in reversed(range(0, data_len)):
            day_index_reversed = data_len - day_loop - 1  # day_loop 倒数的 index
            if (day_index_reversed < self.min_wave_width_left):
                continue
            if (day_loop >= self.min_wave_width_right):
                # 计算 middle_value 是否是波峰或波谷
                middle_value = pp_data[day_loop][PPI_close]
                is_peak = True
                is_valley = True
                for iloop in range(1, self.min_wave_width_left + 1):
                    temp_value = pp_data[day_loop + iloop][PPI_close]
                    if middle_value > temp_value:
                        is_valley = False
                    elif middle_value < temp_value:
                        is_peak = False
                for iloop in range(1, self.min_wave_width_right + 1):
                    temp_value = pp_data[day_loop - iloop][PPI_close]
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
            else:
                extreme_flag = EXTREME_NONE

            # 计算 current_status 当天波动状态
            if extreme_flag != EXTREME_NONE:
                last_extreme = extreme_flag
                current_status = STATUS_NONE
                if extreme_count == 0:
                    self.FillFirstWaveData(extreme_flag, day_loop + 1)
                extreme_count += 1
            else:
                if last_extreme == EXTREME_PEAK:
                    current_status = STATUS_DOWN
                elif last_extreme == EXTREME_VALLEY:
                    current_status = STATUS_UP
                else:
                    current_status = STATUS_NONE

            # 对 input_pp_data 赋值
            self.wave_extreme[day_loop] = extreme_flag
            self.wave_status[day_loop] = current_status

        
    def TradeNextStatus(self, pp_data, day_index, on_day_index):
        wave_extreme = self.wave_extreme[day_index]
        wave_status = self.wave_status[day_index]
        close = pp_data[day_index][PPI_close]
        trade_flag = TS_NONE
        if on_day_index == INVALID_INDEX:
            on_price = 0.0
        else:
            on_price = pp_data[on_day_index][PPI_open]
        if self.last_peak > 0.0 and self.last_valley > 0.0:
            close_10_avg = pp_data[day_index][PPI_close_10_avg]
            close_100_avg = pp_data[day_index][PPI_close_100_avg]
            if (close > self.last_peak) and \
                 ((not self.up_100avg_condition) or (close_10_avg > close_100_avg)) and \
                 (self.trade_off_count > self.trade_off_threshold):
                trade_flag = TS_ON
            elif (wave_extreme == EXTREME_PEAK and close <= self.last_peak) or \
                 (wave_extreme == EXTREME_VALLEY and close <= self.last_valley) or \
                 (wave_status == STATUS_DOWN and close < (on_price * (1.0 - self.cut_loss_ratio))) or \
                 (close < self.last_valley):
                trade_flag = TS_OFF

        if on_day_index != INVALID_INDEX:
            self.trade_off_count = 0
        else:
            self.trade_off_count += 1

        if wave_extreme == EXTREME_PEAK:
            self.last_peak = close
        elif wave_extreme == EXTREME_VALLEY:
            self.last_valley = close
        return trade_flag

def main(argv):
    del argv

    o_data_source = tushare_data.DataSource(20000101, '', '', 1, 20120101, 20200106, False, False, True)
    o_feature = feature.Feature(30, feature.FUT_D5_NORM, 1, False, False)
    # o_feature = feature.Feature(30, feature.FUT_D5_NORM, 1, False, False)
    # o_feature = feature.Feature(30, feature.FUT_5REGION5_NORM, 5, False, False)
    # o_feature = feature.Feature(30, feature.FUT_2AVG5_NORM, 5, False, False)
    o_wave = ExtremeWave(o_data_source, o_feature, 2, 2, True, 0, 0.1, 5)
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
        o_wave.TradeTestStock(FLAGS.c)
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
    flags.DEFINE_integer('epoch', 1000, 'train or rtest epoch')
    # flags.DEFINE_boolean('testall', False, 'test all stocks, save dataset')
    app.run(main)
    