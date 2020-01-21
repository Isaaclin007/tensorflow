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
import dsfa3d_dataset
import dqn_test

reload(sys)
sys.setdefaultencoding('utf-8')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

FLAGS = flags.FLAGS

class DQNFix(trade_base.TradeBase):
    def __init__(self,
                 o_data_source,
                 o_feature,
                 label_days = 10,
                 decay_ratio = 0.8):
        class_name = 'dqn_fix'
        self.data_source = o_data_source
        self.feature = o_feature
        self.dataset_size = 0
        self.dataset_len = 0
        self.label_days = label_days
        self.decay_ratio = decay_ratio
        app_setting_name = '%u_%.4f' % (label_days, decay_ratio)
        super(DQNFix, self).__init__(o_data_source, 
                                      o_feature, 
                                      class_name, 
                                      app_setting_name,
                                      -100.0,
                                      1)
        self.dsfa = dsfa3d_dataset.DSFa3DDataset(o_data_source, o_feature)

    def CreateDataSet(self):
        dataset_file_name = self.FileNameDataset()
        if os.path.exists(dataset_file_name):
            print('dataset already exist: %s' % dataset_file_name)
            return

        dqn_src_dataset = self.dsfa.GetDSFa3DDataset()
        print("dqn_src_dataset: {}".format(dqn_src_dataset.shape))

        date_num = dqn_src_dataset.shape[0]
        code_num = dqn_src_dataset.shape[1]
        data_unit_date_index = self.feature.index_date
        data_unit_open_index = self.feature.index_open
        data_unit_tscode_index = self.feature.index_tscode
        feature_size = self.feature.feature_size

        # feature | label(从feature_date+1开始计算) | feature_date | ts_code
        dataset = np.zeros((date_num * code_num, feature_size + self.acture_size))
        data_num = 0
        # 允许的最大 trade date 数量（含停牌）
        max_label_date_num = self.label_days * 2
        
        for code_index in range(0, code_num):
            ts_code = dqn_src_dataset[0][code_index][data_unit_tscode_index]
            for day_loop in reversed(range(0, date_num)):
                pre_on_date = dqn_src_dataset[day_loop][code_index][data_unit_date_index]
                # if temp_date > dqn_dataset.dataset_train_test_split_date:
                #     break
                if (pre_on_date > 0):
                    temp_price_ratio = 1.0
                    temp_count = 0
                    temp_date_count = 0
                    temp_effect_ratio = 1.0
                    temp_index = day_loop - 1
                    # 从 T1 open 开始计算 increase
                    # if temp_index < 0:
                    #     continue
                    # on_date = dqn_src_dataset[temp_index][code_index][data_unit_date_index]
                    # current_price = dqn_src_dataset[temp_index][code_index][data_unit_open_index]
                    # temp_index -= 1
                    temp_status = TS_PRE_ON
                    while(temp_index >= 0):
                        temp_date = dqn_src_dataset[temp_index][code_index][data_unit_date_index]
                        if temp_date > 0:
                            temp_open = dqn_src_dataset[temp_index][code_index][data_unit_open_index]
                            if temp_status == TS_PRE_ON:
                                on_date = temp_date
                                current_price = temp_open
                                temp_status = TS_ON
                            elif temp_status == TS_ON:
                                temp_price = temp_open
                                temp_increase = temp_price / current_price - 1.0
                                temp_price_ratio *= (temp_increase * temp_effect_ratio + 1.0)
                                temp_effect_ratio *= self.decay_ratio
                                current_price = temp_price
                                temp_count += 1
                                if temp_count == self.label_days:
                                    temp_status = TS_OFF
                                    off_date = temp_date
                                    break
                        temp_index -= 1
                        temp_date_count += 1
                        if temp_date_count > max_label_date_num:
                            break
                    if temp_status == TS_OFF:
                        temp_label = temp_price_ratio - 1.0
                        dataset[data_num][:feature_size] = dqn_src_dataset[day_loop][code_index][:feature_size]
                        dataset[data_num][self.index_increase] = temp_label
                        dataset[data_num][self.index_ts_code] = ts_code
                        dataset[data_num][self.index_pre_on_date] = pre_on_date
                        dataset[data_num][self.index_on_date] = on_date
                        dataset[data_num][self.index_pre_off_date] = off_date
                        dataset[data_num][self.index_off_date] = off_date
                        dataset[data_num][self.index_holding_days] = self.label_days
                        data_num += 1
            print("%-4d : %06.0f 100%%" % (code_index, ts_code))
        dataset = dataset[:data_num]

        print("dataset: {}".format(dataset.shape))
        print("file_name: %s" % dataset_file_name)
        np.save(dataset_file_name, dataset)

    def ShowDataSet(self, dataset, caption):
        for iloop in range(len(dataset)):
            print("\n%s[%u]:" % (caption, iloop))
            print("-" * 80),
            for dloop in range(dataset.shape[1]):
                if dloop % 5 == 0:
                    print("")
                print("%-16.4f" % dataset[iloop][dloop]),
            print("")
        print("\n")

def main(argv):
    del argv

    o_data_source = tushare_data.DataSource(20000101, '', '', 10, 20000101, 20200106, False, False, True)
    # o_feature = feature.Feature(30, feature.FUT_D5_NORM, 1, False, False)
    # o_feature = feature.Feature(30, feature.FUT_5REGION5_NORM, 5, False, False)
    o_feature = feature.Feature(100, feature.FUT_D5_NORM, 1, False, False)
    o_dqn_fix = DQNFix(o_data_source, o_feature, 50, 0.95)
    # o_dqn_fix = DQNFix(o_data_source, o_feature, 10, 0.8)
    # split_date = 20180101
    split_date = 20100101
    o_dl_model = dl_model.DLModel('%s_%u' % (o_dqn_fix.setting_name, split_date), 
                                o_feature.feature_unit_num, 
                                o_feature.feature_unit_size,
                                32, 10240, 0.04, 'mean_absolute_tp0_max_ratio_error')
    o_dqn_test = dqn_test.DQNTest(o_dqn_fix.dsfa, split_date, o_dl_model)
    if FLAGS.mode == 'data':
        o_data_source.DownloadData()
        o_data_source.UpdatePPData()
    elif FLAGS.mode == 'dataset':
        o_dqn_fix.CreateDataSet()
    elif FLAGS.mode == 'train':
        tf, tl, vf, vl, td = o_dqn_fix.GetDataset(split_date)
        # tf, tl, vf, vl, va = o_dqn_fix.GetDatasetRandom(0.5)
        tl = tl * 100.0
        vl  = vl * 100.0
        o_dl_model.Train(tf, tl, vf, vl, FLAGS.epoch)
    elif FLAGS.mode == 'rtest':
        # tf, tl, vf, vl, va = o_dqn_fix.GetDataset(split_date)
        tf, tl, vf, vl, va = o_dqn_fix.GetDatasetRandom(0.5)
        o_dl_model.LoadModel(FLAGS.epoch)
        o_dqn_fix.RTest(o_dl_model, vf, va, False)
    elif FLAGS.mode == 'dqntest':
        o_dl_model.LoadModel(FLAGS.epoch)
        o_dqn_test.Test(True)
    elif FLAGS.mode == 'dsw':
        dataset = o_dqn_fix.ShowDSW3DDataset()
    elif FLAGS.mode == 'show':
        dataset = o_dqn_fix.ShowTradePP(FLAGS.c)
    elif FLAGS.mode == 'debug':
        dataset = np.load(o_dqn_fix.FileNameDataset())
        print("dataset: {}".format(dataset.shape))
        dataset = np_common.Sort2D(dataset, [o_dqn_fix.index_increase], [False])
        dataset = dataset[:5]
        o_dqn_fix.ShowDataSet(dataset, 'dataset')
    elif FLAGS.mode == 'clean':
        o_dqn_fix.Clean()
        o_dl_model.Clean()

    exit()

if __name__ == "__main__":
    flags.DEFINE_string('mode', 'test', 'test | testall | train')
    flags.DEFINE_string('c', '000001.SZ', 'ts code')
    flags.DEFINE_integer('epoch', 100, 'train or rtest epoch')
    flags.DEFINE_boolean('show', False, 'show trade record')
    app.run(main)
    