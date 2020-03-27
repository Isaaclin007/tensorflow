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

DECAY_MODE_EXP = 0
DECAY_MODE_LINE = 1

class DQNFix(trade_base.TradeBase):
    def __init__(self,
                 o_data_source,
                 o_feature,
                 label_days = 10,
                 label_times_vol_ratio = False,
                 decay_mode = DECAY_MODE_EXP,
                 decay_ratio = 0.8,
                 no_overlap_feature = True):
        class_name = 'dqn_fix'
        self.data_source = o_data_source
        self.feature = o_feature
        self.dataset_size = 0
        self.dataset_len = 0
        self.label_days = label_days
        self.label_times_vol_ratio = label_times_vol_ratio
        self.decay_mode = decay_mode
        self.decay_ratio = decay_ratio
        self.no_overlap_feature = no_overlap_feature

        app_setting_name = '%u_%u_%.4f_%u' % (label_days, decay_mode, decay_ratio, no_overlap_feature)
        if label_times_vol_ratio:
            app_setting_name = '%s_%u' % (app_setting_name, label_times_vol_ratio)
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
            if self.no_overlap_feature:
                start_index = random.randint(0, self.feature.feature_unit_num-1)
                day_list = reversed(range(start_index, date_num, self.feature.feature_unit_num))
            else:
                day_list = reversed(range(0, date_num))
            for day_loop in day_list:
                pre_on_date = dqn_src_dataset[day_loop][code_index][data_unit_date_index]
                # if temp_date > dqn_dataset.dataset_train_test_split_date:
                #     break
                if (pre_on_date > 0):
                    temp_price_ratio = 1.0
                    temp_count = 0
                    temp_date_count = 0
                    temp_effect_ratio = 1.0
                    temp_index = day_loop
                    temp_status = TS_PRE_ON
                    while(1):
                        temp_index -= 1
                        if temp_index < 0:
                            break
                        temp_date = dqn_src_dataset[temp_index][code_index][data_unit_date_index]
                        if temp_date > 0:
                            temp_open = dqn_src_dataset[temp_index][code_index][data_unit_open_index]
                            if temp_status == TS_PRE_ON:
                                # T+1 日状态切换为 TS_ON
                                on_date = temp_date
                                current_price = temp_open
                                temp_status = TS_ON
                                if self.label_times_vol_ratio:
                                    vol = dqn_src_dataset[temp_index][code_index][self.feature.index_vol]
                                    vol_100_avg = dqn_src_dataset[temp_index][code_index][self.feature.index_vol_100_avg]
                                    vol_threshold = vol_100_avg * 0.2  # 相对值
                                    # vol_threshold = 10000  # 绝对值
                                    if vol < vol_threshold:
                                        temp_price_ratio *= vol / vol_threshold
                                # if self.decay_mode == DECAY_MODE_LINE:
                                #     # 开盘涨幅低的优先
                                #     open_increase = dqn_src_dataset[temp_index][code_index][self.feature.index_open_increase]
                                #     open_increase *= 0.01
                                #     temp_price_ratio -= open_increase
                            elif temp_status == TS_ON:
                                temp_price = temp_open
                                temp_increase = temp_price / current_price - 1.0
                                temp_price_ratio *= (temp_increase * temp_effect_ratio + 1.0)
                                if self.decay_mode == DECAY_MODE_EXP:
                                    temp_effect_ratio *= self.decay_ratio
                                elif self.decay_mode == DECAY_MODE_LINE:
                                    temp_effect_ratio -= self.decay_ratio
                                    if temp_effect_ratio < 0.0:
                                        temp_effect_ratio = 0.0
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

    code_filter = '\
000001.SZ,\
000002.SZ,\
000005.SZ,\
000006.SZ,\
000009.SZ,\
000012.SZ,\
000016.SZ,\
000021.SZ,\
000027.SZ,\
000031.SZ,\
000036.SZ,\
000039.SZ,\
000050.SZ,\
000059.SZ,\
000060.SZ,\
000063.SZ,\
000066.SZ,\
000069.SZ,\
000078.SZ,\
000088.SZ,\
000089.SZ,\
000401.SZ,\
000402.SZ,\
000410.SZ,\
000413.SZ,\
000422.SZ,\
000425.SZ,\
000503.SZ,\
000507.SZ,\
000510.SZ,\
000518.SZ,\
000520.SZ,\
000528.SZ,\
000540.SZ,\
000554.SZ,\
000559.SZ,\
000563.SZ,\
000571.SZ,\
000572.SZ,\
000592.SZ,\
000598.SZ,\
000601.SZ,\
000616.SZ,\
000625.SZ,\
000627.SZ,\
000629.SZ,\
000630.SZ,\
000636.SZ,\
000650.SZ,\
000651.SZ,\
000652.SZ,\
000656.SZ,\
000659.SZ,\
000667.SZ,\
000670.SZ,\
000680.SZ,\
000682.SZ,\
000683.SZ,\
000686.SZ,\
000690.SZ,\
000709.SZ,\
000717.SZ,\
000718.SZ,\
000720.SZ,\
000723.SZ,\
000727.SZ,\
000728.SZ,\
000735.SZ,\
000750.SZ,\
000751.SZ,\
000758.SZ,\
000767.SZ,\
000768.SZ,\
000776.SZ,\
000778.SZ,\
000783.SZ,\
000789.SZ,\
000793.SZ,\
000795.SZ,\
000800.SZ,\
000806.SZ,\
000807.SZ,\
000816.SZ,\
000822.SZ,\
000823.SZ,\
000825.SZ,\
000829.SZ,\
000830.SZ,\
000831.SZ,\
000839.SZ,\
000851.SZ,\
000858.SZ,\
000859.SZ,\
000868.SZ,\
000876.SZ,\
000877.SZ,\
000878.SZ,\
000882.SZ,\
000886.SZ,\
000897.SZ,\
000898.SZ,\
000917.SZ,\
000927.SZ,\
000930.SZ,\
000931.SZ,\
000932.SZ,\
000933.SZ,\
000937.SZ,\
000959.SZ,\
600000.SH,\
600006.SH,\
600060.SH,\
600063.SH,\
600067.SH,\
600068.SH,\
600069.SH,\
600078.SH,\
600089.SH,\
600100.SH,\
600103.SH,\
600104.SH,\
600108.SH,\
600109.SH,\
600110.SH,\
600111.SH,\
600115.SH,\
600122.SH,\
600123.SH,\
600125.SH,\
600127.SH,\
600151.SH,\
600153.SH,\
600157.SH,\
600158.SH,\
600160.SH,\
600162.SH,\
600166.SH,\
600169.SH,\
600170.SH,\
600171.SH,\
600175.SH,\
600176.SH,\
600177.SH,\
600183.SH,\
600186.SH,\
600187.SH,\
600188.SH,\
600196.SH,\
600198.SH,\
600200.SH,\
600206.SH,\
600208.SH,\
600209.SH,\
600210.SH,\
600212.SH,\
600216.SH,\
600219.SH,\
600220.SH,\
600221.SH,\
600226.SH,\
600239.SH,\
600266.SH,\
600601.SH,\
600606.SH,\
600609.SH,\
600611.SH,\
600624.SH,\
600630.SH,\
600635.SH,\
600642.SH,\
600643.SH,\
600649.SH,\
600651.SH,\
600652.SH,\
600653.SH,\
600660.SH,\
600662.SH,\
600664.SH,\
600667.SH,\
600673.SH,\
600674.SH,\
600675.SH,\
600677.SH,\
600688.SH,\
600690.SH,\
600703.SH,\
600704.SH,\
600705.SH,\
600711.SH,\
600717.SH,\
600718.SH,\
600720.SH,\
600736.SH,\
600737.SH,\
600739.SH,\
600740.SH,\
600741.SH,\
600755.SH,\
600759.SH,\
600770.SH,\
600776.SH,\
600777.SH,\
600782.SH,\
600787.SH,\
600789.SH,\
600795.SH,\
600797.SH,\
600800.SH,\
600804.SH,\
600805.SH,\
600808.SH,\
600811.SH,\
600812.SH,\
600816.SH,\
600820.SH,\
600837.SH,\
600839.SH,\
600846.SH,\
600851.SH,\
600863.SH,\
600868.SH,\
600871.SH,\
600872.SH,\
600873.SH,\
600874.SH,\
600875.SH,\
600877.SH,\
600879.SH,\
600881.SH,\
600886.SH,\
600887.SH,\
600895.SH,\
601607.SH\
'

    # end_date = 20200106
    end_date = 20200306
    # end_date = 20190221
    split_date = 20100101
    o_data_source = tushare_data.DataSource(20000101, '', code_filter, 1, 20000101, end_date, False, False, True)
    # o_feature = feature.Feature(30, feature.FUT_D5_NORM, 1, False, False)
    o_feature = feature.Feature(7, feature.FUT_D5_NORM, 1, False, False)
    # o_feature = feature.Feature(7, feature.FUT_5REGION5_NORM, 5, False, False)
    # o_feature = feature.Feature(30, feature.FUT_5REGION5_NORM, 5, False, False)
    # o_feature = feature.Feature(30, feature.FUT_D3_NORM, 1, False, False)
    # o_dqn_fix = DQNFix(o_data_source, o_feature, 6, DECAY_MODE_EXP, 0.6, not FLAGS.overlap_feature)
    o_dqn_fix = DQNFix(o_data_source, o_feature, 6, False, DECAY_MODE_EXP, 0.6, not FLAGS.overlap_feature)
    # o_dqn_fix = DQNFix(o_data_source, o_feature, 30, DECAY_MODE_EXP, 0.9, not FLAGS.overlap_feature)
    o_dl_model = dl_model.DLModel('%s_%u' % (o_dqn_fix.setting_name, split_date), 
                                o_feature.feature_unit_num, 
                                o_feature.feature_unit_size,
                                # 32, 10240, 0.04, 'mean_absolute_tp0_max_ratio_error') # rtest<0
                                # 4, 10240, 0.04, 'mean_absolute_tp0_max_ratio_error') # rtest<0
                                # 4, 10240, 0.01, 'mean_absolute_tp0_max_ratio_error') # rtest:0.14
                                0, 10240, 0.03, 'mean_absolute_tp_max_ratio_error_tanhmap', 50) # rtest:0.62
                                # 16, 10240, 0.01, 'mean_absolute_tp0_max_ratio_error') # rtest<0
                                # 16, 10240, 0.01, 'mean_absolute_tp_max_ratio_error_tanhmap', 100)
    if FLAGS.mode == 'datasource':
        o_data_source.DownloadData()
        o_data_source.UpdatePPData()
    elif FLAGS.mode == 'dataset':
        o_dqn_fix.CreateDataSet()
    elif FLAGS.mode == 'public_dataset':
        o_dqn_fix.CreateDataSet()
        public_dataset = o_dqn_fix.PublicDataset()
        file_name = './public/data/dataset.npy'
        np.save(file_name, public_dataset)
    elif FLAGS.mode == 'train':
        tf, tl, vf, vl, td = o_dqn_fix.GetDataset(split_date)
        # tf, tl, vf, vl, va = o_dqn_fix.GetDatasetRandom(0.5)
        train_epoch = FLAGS.epoch if FLAGS.epoch > 0 else 250
        o_dl_model.Train(tf, tl, vf, vl, train_epoch)
    elif FLAGS.mode == 'rtest':
        tf, tl, vf, vl, va = o_dqn_fix.GetDataset(split_date)
        # tf, tl, vf, vl, va = o_dqn_fix.GetDatasetRandom(0.5)
        o_dl_model.LoadModel(FLAGS.epoch)
        o_dqn_fix.RTest(o_dl_model, vf, va, False)
    elif FLAGS.mode == 'dqntest':
        o_dl_model.LoadModel(FLAGS.epoch)
        o_dsfa = dsfa3d_dataset.DSFa3DDataset(o_data_source, o_feature)
        o_dqn_test = dqn_test.DQNTest(o_dsfa, split_date, o_dl_model)
        o_dqn_test.Test(1, FLAGS.pt, True, FLAGS.show)
    elif FLAGS.mode == 'dqntestall':
        o_dl_model.LoadModel(FLAGS.epoch)
        o_dsfa = dsfa3d_dataset.DSFa3DDataset(o_data_source, o_feature)
        o_dqn_test = dqn_test.DQNTest(o_dsfa, split_date, o_dl_model)
        o_dqn_test.TestAllModels(1, FLAGS.pt)
    elif FLAGS.mode == 'predict':
        o_dl_model.LoadModel(FLAGS.epoch)
        o_data_source.SetPPDataDailyUpdate(20180101, 20200323)
        o_dsfa = dsfa3d_dataset.DSFa3DDataset(o_data_source, o_feature)
        o_dqn_test = dqn_test.DQNTest(o_dsfa, split_date, o_dl_model)
        o_dqn_test.Test(1, FLAGS.pt, True, FLAGS.show)
    elif FLAGS.mode == 'dsw':
        dataset = o_dqn_fix.ShowDSW3DDataset()
    elif FLAGS.mode == 'show':
        dataset = o_dqn_fix.ShowTradePP(FLAGS.c)
    elif FLAGS.mode == 'showlabel':
        dataset = o_dqn_fix.ShowLabel()
    elif FLAGS.mode == 'debug':
        dataset = np.load(o_dqn_fix.FileNameDataset())
        print("dataset: {}".format(dataset.shape))
        dataset = np_common.Sort2D(dataset, [o_dqn_fix.index_increase], [False])
        dataset = dataset[:5]
        o_dqn_fix.ShowDataSet(dataset, 'dataset')
    elif FLAGS.mode == 'clean':
        o_dqn_fix.Clean()
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
    