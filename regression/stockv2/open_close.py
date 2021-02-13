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

class OpenClose(trade_base.TradeBase):
    def __init__(self,
                 o_data_source,
                 o_feature,
                 no_overlap_feature = True):
        class_name = 'open_close'
        self.data_source = o_data_source
        self.feature = o_feature
        self.dataset_size = 0
        self.dataset_len = 0
        self.no_overlap_feature = no_overlap_feature

        app_setting_name = '%u' % (no_overlap_feature)
        super(OpenClose, self).__init__(o_data_source, 
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
        feature_size = self.feature.feature_size

        # feature | label(从feature_date+1开始计算) | feature_date | ts_code
        dataset = np.zeros((date_num * code_num, feature_size + self.acture_size))
        data_num = 0
        # 允许的最大 trade date 数量（含停牌）
        
        for code_index in range(0, code_num):
            ts_code = dqn_src_dataset[0][code_index][self.feature.index_tscode]
            if self.no_overlap_feature:
                start_index = random.randint(0, self.feature.feature_unit_num-1)
                day_list = reversed(range(start_index, date_num, self.feature.feature_unit_num))
            else:
                day_list = reversed(range(0, date_num))
            for day_loop in day_list:
                pre_on_date = dqn_src_dataset[day_loop][code_index][self.feature.index_date]
                # if temp_date > dqn_dataset.dataset_train_test_split_date:
                #     break
                if (pre_on_date > 0):
                    label_index = day_loop - 1
                    if label_index >= 0:
                        label_date = dqn_src_dataset[label_index][code_index][self.feature.index_date]
                        if label_date > 0:
                            dataset[data_num][:feature_size] = dqn_src_dataset[day_loop][code_index][:feature_size]
                            temp_open = dqn_src_dataset[label_index][code_index][self.feature.index_open]
                            temp_close = dqn_src_dataset[label_index][code_index][self.feature.index_close]
                            dataset[data_num][self.index_increase] = temp_close / temp_open - 1.0
                            dataset[data_num][self.index_ts_code] = ts_code
                            dataset[data_num][self.index_pre_on_date] = pre_on_date
                            dataset[data_num][self.index_on_date] = label_date
                            dataset[data_num][self.index_pre_off_date] = label_date
                            dataset[data_num][self.index_off_date] = label_date
                            dataset[data_num][self.index_holding_days] = 1
                            data_num += 1
            print("dqn_fix - %-4d : %06.0f 100%%" % (code_index, ts_code))
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

    end_date = 20200306
    split_date = 20100101
    o_data_source = tushare_data.DataSource(20000101, '', '', 10, 20000101, end_date, False, False, True)
    o_feature = feature.Feature(10, feature.FUT_D5_NORM, 1, False, False)
    obj = OpenClose(o_data_source, o_feature, not FLAGS.overlap_feature)
    o_dl_model = dl_model.DLModel('%s_%u' % (obj.setting_name, split_date), 
                                o_feature.feature_unit_num, 
                                o_feature.feature_unit_size,
                                # 32, 10240, 0.04, 'mean_absolute_tp0_max_ratio_error') # rtest<0
                                # 4, 10240, 0.04, 'mean_absolute_tp0_max_ratio_error') # rtest<0
                                # 4, 10240, 0.01, 'mean_absolute_tp0_max_ratio_error') # rtest:0.14
                                32, 10240, 0.03, 'mean_absolute_tp_max_ratio_error_tanhmap', 50) # rtest:0.62
                                # 16, 10240, 0.01, 'mean_absolute_tp0_max_ratio_error') # rtest<0
                                # 16, 10240, 0.01, 'mean_absolute_tp_max_ratio_error_tanhmap', 100)
    if FLAGS.mode == 'datasource':
        o_data_source.DownloadData()
        o_data_source.UpdatePPData()
    elif FLAGS.mode == 'dataset':
        obj.CreateDataSet()
    elif FLAGS.mode == 'public_dataset':
        obj.CreateDataSet()
        public_dataset = obj.PublicDataset()
        file_name = './public/data/dataset.npy'
        np.save(file_name, public_dataset)
    elif FLAGS.mode == 'train':
        tf, tl, vf, vl, td = obj.GetDataset(split_date)
        # tf, tl, vf, vl, va = obj.GetDatasetRandom(0.5)
        train_epoch = FLAGS.epoch if FLAGS.epoch > 0 else 250
        o_dl_model.Train(tf, tl, vf, vl, train_epoch)
    elif FLAGS.mode == 'rtest':
        tf, tl, vf, vl, va = obj.GetDataset(split_date)
        # tf, tl, vf, vl, va = obj.GetDatasetRandom(0.5)
        o_dl_model.LoadModel(FLAGS.epoch)
        obj.RTest(o_dl_model, vf, va, False)
    # elif FLAGS.mode == 'dqntest':
    #     o_dl_model.LoadModel(FLAGS.epoch)
    #     o_dsfa = dsfa3d_dataset.DSFa3DDataset(o_data_source, o_feature)
    #     o_dqn_test = dqn_test.DQNTest(o_dsfa, split_date, o_dl_model)
    #     o_dqn_test.Test(1, FLAGS.pt, True, FLAGS.show)
    # elif FLAGS.mode == 'dqntestall':
    #     o_dl_model.LoadModel(FLAGS.epoch)
    #     o_dsfa = dsfa3d_dataset.DSFa3DDataset(o_data_source, o_feature)
    #     o_dqn_test = dqn_test.DQNTest(o_dsfa, split_date, o_dl_model)
    #     o_dqn_test.TestAllModels(1, FLAGS.pt)
    # elif FLAGS.mode == 'predict':
    #     o_dl_model.LoadModel(FLAGS.epoch)
    #     o_data_source.SetPPDataDailyUpdate(20180101, 20200323)
    #     o_dsfa = dsfa3d_dataset.DSFa3DDataset(o_data_source, o_feature)
    #     o_dqn_test = dqn_test.DQNTest(o_dsfa, split_date, o_dl_model)
    #     o_dqn_test.Test(1, FLAGS.pt, True, FLAGS.show)
    elif FLAGS.mode == 'dsw':
        dataset = obj.ShowDSW3DDataset()
    elif FLAGS.mode == 'show':
        dataset = obj.ShowTradePP(FLAGS.c)
    elif FLAGS.mode == 'showlabel':
        dataset = obj.ShowLabel()
    elif FLAGS.mode == 'debug':
        dataset = np.load(obj.FileNameDataset())
        print("dataset: {}".format(dataset.shape))
        dataset = np_common.Sort2D(dataset, [obj.index_increase], [False])
        dataset = dataset[:5]
        obj.ShowDataSet(dataset, 'dataset')
    elif FLAGS.mode == 'clean':
        obj.Clean()
        o_dl_model.Clean()
    elif FLAGS.mode == 'pp':
        o_data_source.ShowStockPPData(FLAGS.c, FLAGS.date)
    elif FLAGS.mode == 'vol':
        o_data_source.ShowAvgVol(100000)

    exit()

if __name__ == "__main__":
    flags.DEFINE_string('mode', 'test', 'test | testall | train')
    flags.DEFINE_string('c', '000001.SZ', 'ts code')
    flags.DEFINE_integer('epoch', -1, 'train or rtest epoch')
    flags.DEFINE_boolean('show', False, 'show trade record')
    flags.DEFINE_boolean('overlap_feature', True, 'overlap featrue')
    flags.DEFINE_integer('date', 20000101, 'trade date')
    flags.DEFINE_integer('pt', 5, 'predict threshold')
    app.run(main)
    