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
                 predict_threshold,
                 dataset_sample_num,
                 cut_loss_ratio = 1.0):
        self.data_source = o_data_source
        self.feature = o_feature
        self.dataset_size = 0
        self.dataset_len = 0
        self.class_name = class_name
        self.app_setting_name = app_setting_name
        self.dataset_sample_num = dataset_sample_num
        self.predict_threshold = predict_threshold
        self.cut_loss_ratio = cut_loss_ratio
        self.setting_name = '%s_%s_%s_%s_%u_%.2f' % (class_name, 
                                                app_setting_name, 
                                                o_data_source.setting_name,
                                                o_feature.setting_name,
                                                dataset_sample_num,
                                                cut_loss_ratio)
        
        self.offset_increase     = 0
        self.offset_ts_code      = 1
        self.offset_pre_on_date  = 2
        self.offset_on_date      = 3
        self.offset_pre_off_date = 4
        self.offset_off_date     = 5
        self.offset_holding_days = 6
        self.acture_size = 7
        self.index_increase     = self.feature.feature_size
        self.index_ts_code      = self.feature.feature_size + 1
        self.index_pre_on_date  = self.feature.feature_size + 2
        self.index_on_date      = self.feature.feature_size + 3
        self.index_pre_off_date = self.feature.feature_size + 4
        self.index_off_date     = self.feature.feature_size + 5
        self.index_holding_days = self.feature.feature_size + 6
        self.rtest_max_holding_num = 4

    def FileNameDataset(self):
        return './data/dataset/%s.npy' % self.setting_name

    def FileNameDSW3DDataset(self):
        return './data/dataset/DSW3D_%s.npy' % self.setting_name
        
    def AppendDataUnit(self, data_unit):
        if self.dataset_len >= self.dataset_size:
            self.dataset_size += 1000000
            new_dataset = np.zeros((self.dataset_size, self.feature.feature_size + self.acture_size))
            if self.dataset_len > 0:
                new_dataset[:self.dataset_len] = self.dataset[:]
            self.dataset = new_dataset
        self.dataset[self.dataset_len] = data_unit
        self.dataset_len += 1

    def GetShowData(self, 
                    pp_data,
                    start_index,
                    end_index,
                    max_num = -1,
                    ppi_index = PPI_close):
        data_len = len(pp_data)
        if start_index >= 0 and end_index >= 0:
            p1 = start_index
            p2 = end_index
        elif start_index >= 0 and end_index < 0:
            p1 = start_index
            if max_num > 0:
                p2 = start_index + max_num
            else:
                p2 = data_len
        elif start_index < 0 and end_index >= 0:
            if max_num > 0:
                p1 = end_index - max_num
            else:
                p1 = 0
            p2 = end_index
        else:
            print('Error: GetShowData')
            return []
        if p1 < 0:
            p1 = 0
        if p2 > data_len:
            p2 = data_len
        show_data = np.zeros((p2 - p1, 2))
        show_data[:, 0] = pp_data[p1:p2, PPI_trade_date]
        show_data[:, 1] = pp_data[p1:p2, ppi_index]
        return show_data

    def TradeRecordShow(self, 
                        pp_data,
                        pre_on_day_index,
                        on_day_index,
                        pre_off_day_index,
                        off_day_index):
        if off_day_index == INVALID_INDEX:
            return
        ################### test ###################
        # temp_data = pp_data[pre_on_day_index:pre_on_day_index+100, PPI_close]
        # m = np.mean(temp_data)
        # rmd = np.sum(np.abs(temp_data - m)) / float(len(temp_data)) / m
        # print('rmd: %f' % rmd)
        ############################################
        data_list = []
        name_list = []

        name_list.append('before')
        data_list.append(self.GetShowData(pp_data, on_day_index, -1, 100))

        name_list.append('on')
        data_list.append(self.GetShowData(pp_data, pre_off_day_index, on_day_index + 1))

        name_list.append('after')
        data_list.append(self.GetShowData(pp_data, -1, pre_off_day_index + 1, 100))

        avg_index1 = pre_off_day_index - 100
        avg_index2 = on_day_index + 100
        name_list.append('30_avg')
        data_list.append(self.GetShowData(pp_data, avg_index1, avg_index2, -1, PPI_close_30_avg))

        name_list.append('100_avg')
        show_data_100_avg = self.GetShowData(pp_data, avg_index1, avg_index2, -1, PPI_close_100_avg)
        data_list.append(show_data_100_avg)

        name_list.append('vol')
        show_data_vol = self.GetShowData(pp_data, avg_index1, avg_index2, -1, PPI_vol)
        close_max = max(show_data_100_avg[:, 1])
        vol_max = max(show_data_vol[:, 1])
        vol_ratio = 1.0 / vol_max * close_max / 2
        show_data_vol[:, 1] *= vol_ratio
        data_list.append(show_data_vol)

        np_common.Show2DData('Trade', data_list, name_list, True)



    def TradeRecord(self, 
                    trade_count,
                    pp_data,
                    pre_on_day_index,
                    on_day_index,
                    pre_off_day_index,
                    off_day_index,
                    print_trade_record,
                    save_data_unit,
                    show_trade_record):
        if pre_on_day_index == INVALID_INDEX:
            print('Error: TradeRecord pre_on_day_index == INVALID_INDEX')
            return
        elif on_day_index == INVALID_INDEX:
            pre_on_date = pp_data[pre_on_day_index][PPI_trade_date]
            on_date = INVALID_DATE
            pre_off_date = INVALID_DATE
            off_date = INVALID_DATE
            increase = 0.0
            holding_days = 0
        elif pre_off_day_index == INVALID_INDEX:
            pre_on_date = pp_data[pre_on_day_index][PPI_trade_date]
            on_date = pp_data[on_day_index][PPI_trade_date]
            pre_off_date = INVALID_DATE
            off_date = INVALID_DATE
            off_price = pp_data[0][PPI_open]
            increase = off_price / pp_data[on_day_index][PPI_open] - 1.0
            holding_days = pre_on_day_index
        elif off_day_index == INVALID_INDEX:
            pre_on_date = pp_data[pre_on_day_index][PPI_trade_date]
            on_date = pp_data[on_day_index][PPI_trade_date]
            pre_off_date = pp_data[pre_off_day_index][PPI_trade_date]
            off_date = INVALID_DATE
            off_price = pp_data[0][PPI_open]
            increase = off_price / pp_data[on_day_index][PPI_open] - 1.0
            holding_days = pre_on_day_index - pre_off_day_index
        else:
            pre_on_date = pp_data[pre_on_day_index][PPI_trade_date]
            on_date = pp_data[on_day_index][PPI_trade_date]
            pre_off_date = pp_data[pre_off_day_index][PPI_trade_date]
            off_date = pp_data[off_day_index][PPI_trade_date]
            off_price = pp_data[off_day_index][PPI_open]
            increase = off_price / pp_data[on_day_index][PPI_open] - 1.0
            holding_days = pre_on_day_index - pre_off_day_index
        ts_code = int(pp_data[0][PPI_ts_code])
        if print_trade_record:
            base_common.PrintTrade(trade_count, ts_code, pre_on_date, on_date, pre_off_date, off_date, 
                                   increase, holding_days)
        if show_trade_record:
            self.TradeRecordShow(pp_data,
                                pre_on_day_index,
                                on_day_index,
                                pre_off_day_index,
                                off_day_index)
        if save_data_unit:
            # sample num
            if pre_off_day_index == INVALID_INDEX:
                sample_num = pre_on_day_index + 1
            else:
                sample_num = pre_on_day_index - pre_off_day_index
            if sample_num > self.dataset_sample_num:
                sample_num = self.dataset_sample_num
            # loop
            for iloop in range(sample_num):
                sample_pre_on_day_index = pre_on_day_index - iloop
                sample_pre_on_date = pp_data[sample_pre_on_day_index][PPI_trade_date]
                sample_on_day_index = sample_pre_on_day_index - 1
                if sample_on_day_index < 0:
                    sample_on_date = INVALID_DATE
                    sample_increase = 0.0
                    sample_holding_days = 0
                else:
                    sample_on_date = pp_data[sample_on_day_index][PPI_trade_date]
                    sample_increase = off_price / pp_data[sample_on_day_index][PPI_open] - 1.0
                    sample_holding_days = holding_days - iloop
                data_unit = []
                if self.feature.AppendFeature(pp_data, sample_pre_on_day_index, data_unit):
                    data_unit.append(float(sample_increase))
                    data_unit.append(float(ts_code))
                    data_unit.append(float(sample_pre_on_date))
                    data_unit.append(float(sample_on_date))
                    data_unit.append(float(pre_off_date))
                    data_unit.append(float(off_date))
                    data_unit.append(float(sample_holding_days))
                    self.AppendDataUnit(data_unit)
            # else:
            #     print('Error: TradeRecord AppendFeature False: %s' % ts_code)
        return increase, holding_days

    def TradePP(self, pp_data):
        a = 1

    def TradeNextStatus(self, pp_data, day_index):
        return TS_NONE

    def TradeTest(self, pp_data, 
                  print_trade_record = False, 
                  save_data_unit=False, 
                  show_trade_record = False):
        data_len = len(pp_data)
        if data_len == 0:
            return 0.0, 0, 0
        self.TradePP(pp_data)
        ts_code = int(pp_data[0][PPI_ts_code])

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
            # cut loss
            if trade_status == TS_ON:
                close = pp_data[day_index][PPI_close]
                if close < (on_price * (1.0 - self.cut_loss_ratio)):
                    next_status = TS_OFF

            # OFF -> PRE_ON
            if trade_status == TS_OFF:
                if next_status == TS_ON:
                    trade_status = TS_PRE_ON
                    pre_on_day_index = day_index

            # PRE_ON -> ON
            elif trade_status == TS_PRE_ON:
                trade_status = TS_ON
                on_day_index = day_index
                on_price = pp_data[on_day_index][PPI_open]

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
                                                          save_data_unit,
                                                          show_trade_record)
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
                                                    save_data_unit,
                                                    show_trade_record)

        if print_trade_record:
            base_common.PrintTrade('sum', ts_code, '--', '--', '--', '--', sum_increase, sum_holding_days)
        else:
            code_index = self.data_source.code_index_map_int[ts_code]
            base_common.PrintTrade(trade_count, ts_code, '--', '--', '--', '--', sum_increase, sum_holding_days)
        return sum_increase, trade_count, sum_holding_days

    def TradeTestAll(self, save_data_unit=True, show_trade_record = False):
        sum_increase = 0.0
        sum_trade_count = 0
        sum_holding_days = 0
        for iloop in range(len(self.data_source.code_list)):
            ts_code = self.data_source.code_list[iloop]
            pp_data = self.data_source.LoadStockPPData(ts_code, True)
            increase, trade_count, holding_days = self.TradeTest(pp_data, False, save_data_unit, show_trade_record)
            sum_increase += increase
            sum_trade_count += trade_count
            sum_holding_days += holding_days
        base_common.PrintTrade(sum_trade_count, '--', '--', '--', '--', '--', sum_increase, sum_holding_days)
        if sum_trade_count == 0:
            avg_increase_day = 0.0
            avg_increase_trade = 0.0
        else:
            avg_increase_day = sum_increase / sum_holding_days
            avg_increase_trade = sum_increase / sum_trade_count
        print('avg increase / day : %.4f' % avg_increase_day)
        print('avg increase / trade : %.4f' % avg_increase_trade)
        if self.dataset_len > 0:
            file_name = self.FileNameDataset()
            base_common.MKFileDirs(file_name)
            np.save(file_name, self.dataset[:self.dataset_len])

    def TradeTestStock(self, ts_code, show_trade_record = False):
        pp_data = self.data_source.LoadStockPPData(ts_code, True)
        self.TradeTest(pp_data, True, False, show_trade_record)
        # self.TradeTest(pp_data, False, True)

    def PublicDataset(self):
        file_name = self.FileNameDataset()
        dataset = np.load(file_name)
        print("dataset: {}".format(dataset.shape))
        feature_size = self.feature.feature_size
        public_dataset = np.zeros((len(dataset), feature_size + 2))
        public_dataset[:, 0:feature_size] = dataset[:, 0:feature_size]
        public_dataset[:, feature_size] = dataset[:, self.index_increase] * 100.0
        public_dataset[:, feature_size + 1] = dataset[:, self.index_pre_on_date]
        print("public_dataset: {}".format(public_dataset.shape))
        return public_dataset


    def GetDataset(self, split_date):
        file_name = self.FileNameDataset()
        # file_name = '../stock/data/dataset/wave_dataset_0_30_0_0_20120101_20000101_20000101_20190414___2_2_0_1_0_5_0.npy'
        dataset = np.load(file_name)
        print("dataset: {}".format(dataset.shape))
        pos = dataset[:,self.index_pre_on_date] < split_date
        train_data = dataset[pos]
        test_data = dataset[~pos]

        test_data = np_common.Sort2D(test_data, 
                                     [self.index_pre_on_date, self.index_ts_code])

        print("train: {}".format(train_data.shape))
        print("test: {}".format(test_data.shape))

        train_features = train_data[:, :self.feature.feature_size]
        train_labels = train_data[:, self.feature.feature_size] * 100.0

        test_features = test_data[:, :self.feature.feature_size]
        test_labels = test_data[:, self.feature.feature_size] * 100.0
        test_acture = test_data[:, self.feature.feature_size:]

        return train_features, train_labels, test_features, test_labels, test_acture

    def GetDatasetRandom(self, test_ratio):
        file_name = self.FileNameDataset()
        dataset = np.load(file_name)
        print("dataset: {}".format(dataset.shape))
        data_len = len(dataset)
        test_data_len = int(data_len * test_ratio)
        np.random.seed(0)
        order = np.argsort(np.random.random(data_len))
        train_data = dataset[order[test_data_len:]]
        test_data = dataset[order[:test_data_len]]
        test_data = np_common.Sort2D(test_data, 
                                     [self.index_pre_on_date, self.index_ts_code])

        print("train: {}".format(train_data.shape))
        print("test: {}".format(test_data.shape))

        train_features = train_data[:, :self.feature.feature_size]
        train_labels = train_data[:, self.feature.feature_size] * 100.0

        test_features = test_data[:, :self.feature.feature_size]
        test_labels = test_data[:, self.feature.feature_size] * 100.0
        test_acture = test_data[:, self.feature.feature_size:]

        return train_features, train_labels, test_features, test_labels, test_acture

    def GetDSW3DDataset(self):
        dataset_file_name = self.FileNameDSW3DDataset()
        if not os.path.exists(dataset_file_name):
            dataset = np.zeros((len(self.data_source.date_list), 
                                len(self.data_source.code_list), 
                                1))
            for code_index in range(0, len(self.data_source.code_list)):
                ts_code = self.data_source.code_list[code_index]
                S_index = self.data_source.code_index_map[ts_code]
                pp_data = self.data_source.LoadStockPPData(ts_code, True)
                data_len = len(pp_data)
                if data_len == 0:
                    continue
                self.TradePP(pp_data)
                for iloop in range(data_len):
                    D_index = self.data_source.date_index_map[int(pp_data[iloop][PPI_trade_date])]
                    dataset[D_index][S_index][0] = self.wave_data[iloop]
                sys.stdout.write("%-4d : %s 100%%\n" % (code_index, ts_code))
            base_common.MKFileDirs(dataset_file_name)
            np.save(dataset_file_name, dataset)
        dataset = np.load(dataset_file_name)
        print("dataset: {}".format(dataset.shape))
        return dataset

    def Clean(self):
        dataset_file_name = self.FileNameDataset()
        print(dataset_file_name)
        if os.path.exists(dataset_file_name):
            print('os.remove(%s)' % dataset_file_name)
            os.remove(dataset_file_name)

    def ShowTradePP(self, ts_code):
        pp_data = self.data_source.LoadStockPPData(ts_code, True)
        data_len = len(pp_data)
        if data_len == 0:
            print('Error: ShowTradePP data_len == 0')
        self.TradePP(pp_data)
        data_list = []
        show_data = np.zeros((data_len, 2))
        show_data[:, 0] = pp_data[:, PPI_trade_date]
        show_data[:, 1] = pp_data[:, PPI_close]
        data_list.append(show_data)

        show_data = np.zeros((data_len, 2))
        show_data[:, 0] = pp_data[:, PPI_trade_date]
        show_data[:, 1] = pp_data[:, PPI_close_5_avg]
        data_list.append(show_data)

        up_pp_data = pp_data[self.wave_data == WS_UP]
        show_data = np.zeros((len(up_pp_data), 2))
        show_data[:, 0] = up_pp_data[:, PPI_trade_date]
        show_data[:, 1] = up_pp_data[:, PPI_close]
        data_list.append(show_data)
        np_common.Show2DData('wave_data', data_list, [], True)

    def ShowDSW3DDataset(self):
        dataset = self.GetDSW3DDataset()
        show_data = np.zeros((len(self.data_source.date_list), 2))
        for iloop in range(len(self.data_source.date_list)):
            show_data[iloop][0] = int(self.data_source.date_list[iloop])
            show_data[iloop][1] = np.sum(dataset[iloop, :, 0] == WS_UP)
            print('%-10u%u' % (int(show_data[iloop][0]), int(show_data[iloop][1])))
        index_pp_data = self.data_source.LoadIndexPPData()
        index_show_data = np.zeros((len(index_pp_data), 2))
        index_show_data[:, 0] = index_pp_data[:, PPI_trade_date]
        index_show_data[:, 1] = index_pp_data[:, PPI_close]
        np_common.Show2DData('DSW3D', [show_data, index_show_data], [], True)

    def RTest(self, dl_model, test_features, test_acture, test_features_pretreated=False):
        predictions = dl_model.Predict(test_features, test_features_pretreated)
        pos = predictions.flatten() > self.predict_threshold
        predictions_f = predictions[pos]
        print('predictions_f:{}'.format(predictions_f.shape))
        acture_f = test_acture[pos]
        merge_data = np.hstack((acture_f, predictions_f))
        predictions_index = self.acture_size
        merge_data = np_common.Sort2D(merge_data, 
                                      [self.offset_pre_on_date, predictions_index], 
                                      [True, False])
        pool = []  # 保存 off_date
        trade_count = 0
        sum_increase = 0.0
        sum_holding_days = 0
        for iloop in range(len(merge_data)):
            temp_pre_on = merge_data[iloop][self.offset_pre_on_date]
            temp_off = merge_data[iloop][self.offset_off_date]
            for kloop in range(len(pool)):
                pool_off = pool[kloop]
                if pool_off != INVALID_DATE and pool_off <= temp_pre_on:
                    pool.pop(kloop)
                    break
            if len(pool) < self.rtest_max_holding_num:
                pool.append(temp_off)
                base_common.PrintTrade(trade_count, 
                                        merge_data[iloop][self.offset_ts_code], 
                                        merge_data[iloop][self.offset_pre_on_date], 
                                        merge_data[iloop][self.offset_on_date], 
                                        merge_data[iloop][self.offset_pre_off_date], 
                                        merge_data[iloop][self.offset_off_date], 
                                        merge_data[iloop][self.offset_increase], 
                                        merge_data[iloop][self.offset_holding_days],
                                        merge_data[iloop][predictions_index])
                sum_increase += merge_data[iloop][self.offset_increase] / self.rtest_max_holding_num
                sum_holding_days += merge_data[iloop][self.offset_holding_days]
                trade_count += 1
        base_common.PrintTrade('sum', '--', '--', '--', '--', '--', sum_increase, sum_holding_days, '--')
        if trade_count == 0:
            avg_increase_day = 0.0
            avg_increase_trade = 0.0
        else:
            avg_increase_day = sum_increase * self.rtest_max_holding_num / sum_holding_days
            avg_increase_trade = sum_increase * self.rtest_max_holding_num / trade_count
        print('avg increase / day : %.4f' % avg_increase_day)
        print('avg increase / trade : %.4f' % avg_increase_trade)
            # pos = holding_status[:, self.offset_off_date] <= merge_data[iloop][self.offset_pre_on_date]
            # print('pos:{}'.format(pos))
            # holding_status[pos][self.offset_ts_code] = 0
            # for kloop in range(self.rtest_max_holding_num):
            #     if holding_status[kloop][self.offset_ts_code] == 0 or \
            #        holding_status[kloop][self.offset_off_date] <= merge_data[iloop][self.offset_pre_on_date]:
            #         holding_status[kloop] = merge_data[iloop]
            #         base_common.PrintTrade(trade_count, 
            #                                merge_data[iloop][self.offset_ts_code], 
            #                                merge_data[iloop][self.offset_pre_on_date], 
            #                                merge_data[iloop][self.offset_on_date], 
            #                                merge_data[iloop][self.offset_pre_off_date], 
            #                                merge_data[iloop][self.offset_off_date], 
            #                                merge_data[iloop][self.offset_increase], 
            #                                merge_data[iloop][self.offset_holding_days])
            #         trade_count += 1
            #         break
        
    
# class TradeBaseTest(TradeBase):
#     def __init__(self,
#                  o_data_source,
#                  o_feature,
#                  avg_cycle):
#         class_name = 'TradeBaseTest'
#         app_setting_name = '%u' % avg_cycle
#         self.avg_cycle = avg_cycle
#         self.avg_cycle_name = 'close_%u_avg' % avg_cycle
#         super(TradeBaseTest, self).__init__(o_data_source, o_feature, class_name, app_setting_name)

#     def TradePP(self, pp_data):
#         # Nothing
#         a = 1

#     def TradeNextStatus(self, pp_data, day_index):
#         pre_day_index = day_index + 1
#         if pre_day_index >= len(pp_data):
#             return TS_NONE
#         if pp_data.loc[day_index, self.avg_cycle_name] > pp_data.loc[pre_day_index, self.avg_cycle_name]:
#             return TS_ON
#         else:
#             return TS_OFF

# def main(argv):
#     del argv

#     o_data_source = tushare_data.DataSource(20000101, '', '', 1, 20000101, 20200106, False, False, True)
#     # o_feature = feature.Feature(30, feature.FUT_D5_NORM, 1, False, False)
#     o_feature = feature.Feature(30, feature.FUT_5REGION5_NORM, 1, False, False)
#     o_trade_test = TradeBaseTest(o_data_source, o_feature, 30)
#     split_date = 20170101
#     o_dl_model = dl_model.DLModel('%s_%u' % (o_trade_test.setting_name, split_date), 
#                                   o_feature.feature_unit_num, 
#                                   o_feature.feature_unit_size,
#                                   32, 10240, 0.004, 'mean_absolute_tp0_max_ratio_error')
#     if FLAGS.mode == 'data':
#         o_data_source.DownloadData()
#         o_data_source.UpdatePPData()
#     elif FLAGS.mode == 'testall':
#         o_trade_test.TradeTestAll()
#     elif FLAGS.mode == 'test':
#         o_trade_test.TradeTestStock(FLAGS.c)
#     elif FLAGS.mode == 'train':
#         tf, tl, vf, vl, td = o_trade_test.GetDataset(split_date)
#         o_dl_model.Train(tf, tl, vf, vl, 10)
        
#     exit()

# if __name__ == "__main__":
#     flags.DEFINE_string('mode', 'test', 'test | testall | train')
#     flags.DEFINE_string('c', '000001.SZ', 'ts code')
#     # flags.DEFINE_boolean('testall', False, 'test all stocks, save dataset')
#     app.run(main)
    