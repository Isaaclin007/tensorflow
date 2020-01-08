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

reload(sys)
sys.setdefaultencoding('utf-8')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

FLAGS = flags.FLAGS

class TradeBase(object):
    def __init__(self,
                 o_data_source,
                 o_feature, 
                 class_name,
                 app_setting_name,
                 predict_threshold):
        self.data_source = o_data_source
        self.feature = o_feature
        self.dataset_size = 0
        self.dataset_len = 0
        self.class_name = class_name
        self.app_setting_name = app_setting_name
        self.setting_name = '%s_%s_%s_%s' % (self.class_name, 
                                            self.app_setting_name, 
                                            self.data_source.setting_name,
                                            self.feature.setting_name)
        
        self.predict_threshold = predict_threshold
        self.index_increase     = self.feature.feature_size
        self.index_ts_code      = self.feature.feature_size + 1
        self.index_pre_on_date  = self.feature.feature_size + 2
        self.index_on_date      = self.feature.feature_size + 3
        self.index_pre_off_date = self.feature.feature_size + 4
        self.index_off_date     = self.feature.feature_size + 5
        self.index_holding_days = self.feature.feature_size + 6

    def FileNameDataset(self):
        return './data/dataset/%s.npy' % self.setting_name
        
    def AppendDataUnit(self, data_unit):
        if self.dataset_len >= self.dataset_size:
            self.dataset_size += 1000000
            new_dataset = np.zeros((self.dataset_size, self.feature.feature_size + 7))
            if self.dataset_len > 0:
                new_dataset[:self.dataset_len] = self.dataset[:]
            self.dataset = new_dataset
        self.dataset[self.dataset_len] = data_unit
        self.dataset_len += 1

    def TradeRecord(self, 
                    trade_count,
                    pp_data,
                    pre_on_day_index,
                    on_day_index,
                    pre_off_day_index,
                    off_day_index,
                    print_trade_record,
                    save_data_unit):
        if pre_on_day_index == INVALID_INDEX:
            print('Error: TradeRecord pre_on_day_index == INVALID_INDEX')
            return
        elif on_day_index == INVALID_INDEX:
            pre_on_date = pp_data.loc[pre_on_day_index,'trade_date']
            on_date = INVALID_DATE
            pre_off_date = INVALID_DATE
            off_date = INVALID_DATE
            increase = 0.0
            holding_days = 0
        elif pre_off_day_index == INVALID_INDEX:
            pre_on_date = pp_data.loc[pre_on_day_index,'trade_date']
            on_date = pp_data.loc[on_day_index,'trade_date']
            pre_off_date = INVALID_DATE
            off_date = INVALID_DATE
            increase = pp_data.loc[0, 'open'] / pp_data.loc[on_day_index, 'open'] - 1.0
            holding_days = on_day_index
        elif off_day_index == INVALID_INDEX:
            pre_on_date = pp_data.loc[pre_on_day_index,'trade_date']
            on_date = pp_data.loc[on_day_index,'trade_date']
            pre_off_date = pp_data.loc[pre_off_day_index,'trade_date']
            off_date = INVALID_DATE
            increase = pp_data.loc[0, 'open'] / pp_data.loc[on_day_index, 'open'] - 1.0
            holding_days = on_day_index
        else:
            pre_on_date = pp_data.loc[pre_on_day_index,'trade_date']
            on_date = pp_data.loc[on_day_index,'trade_date']
            pre_off_date = pp_data.loc[pre_off_day_index,'trade_date']
            off_date = pp_data.loc[off_day_index,'trade_date']
            increase = pp_data.loc[off_day_index, 'open'] / pp_data.loc[on_day_index, 'open'] - 1.0
            holding_days = on_day_index - off_day_index
        ts_code = pp_data.loc[0,'ts_code']
        if print_trade_record:
            base_common.PrintTrade(trade_count, ts_code, pre_on_date, on_date, pre_off_date, off_date, 
                                   increase, holding_days)
        if save_data_unit:
            data_unit = []
            if self.feature.AppendFeature(pp_data, pre_on_day_index, data_unit):
                data_unit.append(float(increase))
                data_unit.append(float(ts_code[:6]))
                data_unit.append(float(pre_on_date))
                data_unit.append(float(on_date))
                data_unit.append(float(pre_off_date))
                data_unit.append(float(off_date))
                data_unit.append(float(holding_days))
                self.AppendDataUnit(data_unit)
            # else:
            #     print('Error: TradeRecord AppendFeature False: %s' % ts_code)
        return increase, holding_days

    def TradePP(self, pp_data):
        a = 1

    def TradeNextStatus(self, pp_data, day_index):
        return TS_NONE

    def TradeTest(self, pp_data, print_trade_record = False, save_data_unit=False):
        data_len = len(pp_data)
        if data_len == 0:
            return 0.0
        self.TradePP(pp_data)
        ts_code = pp_data.loc[0, 'ts_code']

        # global status
        trade_count = 0
        sum_increase = 0.0
        capital_ratio = 1.0
        sum_holding_days = 0
        trade_status = TS_OFF

        # trade status
        pre_on_day_index = INVALID_INDEX
        on_day_index = INVALID_INDEX
        pre_off_day_index = INVALID_INDEX
        off_day_index = INVALID_INDEX

        for day_index in reversed(range(0, data_len)):
            next_status = self.TradeNextStatus(pp_data, day_index)

            # OFF -> PRE_ON
            if trade_status == TS_OFF:
                if next_status == TS_ON:
                    trade_status = TS_PRE_ON
                    pre_on_day_index = day_index

            # PRE_ON -> ON
            elif trade_status == TS_PRE_ON:
                trade_status = TS_ON
                on_day_index = day_index

            # ON -> PRE_OFF
            elif trade_status == TS_ON:
                if next_status == TS_OFF:
                    trade_status = TS_PRE_OFF
                    pre_off_day_index = day_index
                
            # PRE_OFF -> OFF
            elif trade_status == TS_PRE_OFF:
                trade_status = TS_OFF
                off_day_index = day_index

                # record
                increase, holding_days = self.TradeRecord(trade_count,
                                                          pp_data,
                                                          pre_on_day_index, 
                                                          on_day_index, 
                                                          pre_off_day_index, 
                                                          off_day_index, 
                                                          print_trade_record,
                                                          save_data_unit)
                sum_increase += increase
                capital_ratio *= (1.0 + increase)
                sum_holding_days += holding_days
                trade_count += 1

                # reset trade status
                pre_on_day_index = INVALID_INDEX
                on_day_index = INVALID_INDEX
                pre_off_day_index = INVALID_INDEX
                off_day_index = INVALID_INDEX
                

        if trade_status != TS_OFF:
            increase, holding_days = self.TradeRecord(trade_count,
                                                    pp_data,
                                                    pre_on_day_index, 
                                                    on_day_index, 
                                                    pre_off_day_index, 
                                                    off_day_index, 
                                                    print_trade_record,
                                                    save_data_unit)
        # if trade_status == TS_PRE_ON:
        #     base_common.PrintTrade(trade_count, ts_code, pre_on_date, INVALID_DATE, INVALID_DATE, INVALID_DATE, '--', '--')
        # elif trade_status == TS_ON:
        #     day_index = 0
        #     off_price = pp_data.loc[day_index, 'open']
        #     increase = off_price / on_price - 1.0
        #     holding_days = on_day_index - day_index
        #     base_common.PrintTrade(trade_count, ts_code, pre_on_date, on_date, INVALID_DATE, INVALID_DATE, increase, holding_days)
        # elif trade_status == TS_PRE_OFF:
        #     day_index = 0
        #     off_price = pp_data.loc[day_index, 'open']
        #     increase = off_price / on_price - 1.0
        #     holding_days = on_day_index - day_index
        #     base_common.PrintTrade(trade_count, ts_code, pre_on_date, on_date, pre_off_date, INVALID_DATE, increase, holding_days)

        if print_trade_record:
            base_common.PrintTrade('sum', ts_code, '--', '--', '--', '--', capital_ratio, sum_holding_days)
        else:
            code_index = self.data_source.code_index_map[ts_code]
            base_common.PrintTrade(code_index, ts_code, '--', '--', '--', '--', capital_ratio, sum_holding_days)
        return sum_increase, sum_holding_days

    def TradeTestAll(self):
        for iloop in range(len(self.data_source.code_list)):
            ts_code = self.data_source.code_list[iloop]
            pp_data = self.data_source.LoadStockPPData(ts_code, True)
            self.TradeTest(pp_data, False, True)
        
        file_name = self.FileNameDataset()
        base_common.MKFileDirs(file_name)
        np.save(file_name, self.dataset[:self.dataset_len])

    def TradeTestStock(self, ts_code):
        pp_data = self.data_source.LoadStockPPData(ts_code, True)
        self.TradeTest(pp_data, True, False)

    def GetDataset(self, split_date):
        file_name = self.FileNameDataset()
        dataset = np.load(file_name)
        print("dataset: {}".format(dataset.shape))
        pos = dataset[:,self.index_pre_on_date] < split_date
        train_data = dataset[pos]
        test_data = dataset[~pos]

        test_data = np_common.Sort2D(test_data, [self.index_pre_on_date, self.index_ts_code])

        print("train: {}".format(train_data.shape))
        print("test: {}".format(test_data.shape))

        train_features = train_data[:, :self.feature.feature_size]
        train_labels = train_data[:, self.feature.feature_size]

        test_features = test_data[:, :self.feature.feature_size]
        test_labels = test_data[:, self.feature.feature_size]
        test_acture = test_data[:, self.feature.feature_size:]

        return train_features, train_labels, test_features, test_labels, test_acture

    def RTest(self, dl_model, test_features, test_acture, test_features_pretreated=False):
        predictions = dl_model.Predict(test_features, test_features_pretreated)
        pos = predictions > self.predict_threshold
        predictions_f = predictions[pos]
        acture_f = test_acture[pos]
        
    
class TradeBaseTest(TradeBase):
    def __init__(self,
                 o_data_source,
                 o_feature,
                 avg_cycle):
        class_name = 'TradeBaseTest'
        app_setting_name = '%u' % avg_cycle
        self.avg_cycle = avg_cycle
        self.avg_cycle_name = 'close_%u_avg' % avg_cycle
        super(TradeBaseTest, self).__init__(o_data_source, o_feature, class_name, app_setting_name)

    def TradePP(self, pp_data):
        # Nothing
        a = 1

    def TradeNextStatus(self, pp_data, day_index):
        pre_day_index = day_index + 1
        if pre_day_index >= len(pp_data):
            return TS_NONE
        if pp_data.loc[day_index, self.avg_cycle_name] > pp_data.loc[pre_day_index, self.avg_cycle_name]:
            return TS_ON
        else:
            return TS_OFF

def main(argv):
    del argv

    o_data_source = tushare_data.DataSource(20000101, '', '', 1, 20000101, 20200106, False, False, True)
    # o_feature = feature.Feature(30, feature.FUT_D5_NORM, 1, False, False)
    o_feature = feature.Feature(30, feature.FUT_5REGION5_NORM, 1, False, False)
    o_trade_test = TradeBaseTest(o_data_source, o_feature, 30)
    split_date = 20170101
    o_dl_model = dl_model.DLModel('%s_%u' % (o_trade_test.setting_name, split_date), 
                                  o_feature.feature_unit_num, 
                                  o_feature.feature_unit_size,
                                  32, 10240, 0.004, 'mean_absolute_tp0_max_ratio_error')
    if FLAGS.mode == 'data':
        o_data_source.DownloadData()
        o_data_source.UpdatePPData()
    elif FLAGS.mode == 'testall':
        o_trade_test.TradeTestAll()
    elif FLAGS.mode == 'test':
        o_trade_test.TradeTestStock(FLAGS.c)
    elif FLAGS.mode == 'train':
        tf, tl, vf, vl, td = o_trade_test.GetDataset(split_date)
        o_dl_model.Train(tf, tl, vf, vl, 10)
        
    exit()

if __name__ == "__main__":
    flags.DEFINE_string('mode', 'test', 'test | testall | train')
    flags.DEFINE_string('c', '000001.SZ', 'ts code')
    # flags.DEFINE_boolean('testall', False, 'test all stocks, save dataset')
    app.run(main)
    