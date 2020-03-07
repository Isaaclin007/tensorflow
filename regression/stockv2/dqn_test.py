# -*- coding:UTF-8 -*-


import numpy as np
import pandas as pd
import os
import time
import datetime
import sys
import math
import random
from absl import app
from absl import flags
sys.path.append("..")
from common import base_common
from common import np_common
from common.const_def import *
import tushare_data
import feature
import dl_model
import trade_base
import dsfa3d_dataset

FLAGS = flags.FLAGS

class DQNTestCodeStatus():
    # DQN Agent
    def __init__(self):
        self.Reset()

    def Reset(self):
        self.status = TS_OFF
        self.ts_code = 0.0
        self.code_index = 0
        self.pre_on_date = INVALID_DATE
        self.on_date = INVALID_DATE
        self.pre_off_date = INVALID_DATE
        self.off_date = INVALID_DATE
        self.on_price = 0.0
        self.off_price = 0.0
        self.current_price = 0.0
        self.holding_days = 0
        self.pre_on_pred = 0.0
        self.current_pred = 0.0
        self.inc = 0.0

    def UpdateInc(self):
        if self.on_price == 0 or self.off_price == 0:
            self.inc = 0.0
        else:
            self.inc = self.off_price / self.on_price - 1.0

    def Print(self, trade_index, sum_increase, capital_ratio):
        print('%-8u%-12s%-12s%-10s%-8.2f%-8.2f%-10s%-10s%-10s%-6u%-8.2f%-8.2f' % (
                trade_index, 
                base_common.TradeDateStr(self.pre_on_date, self.on_date),
                base_common.TradeDateStr(self.pre_off_date, self.off_date),
                base_common.CodeStr(self.ts_code),
                self.pre_on_pred,
                self.current_pred,
                base_common.PriceStr(self.on_price),
                base_common.PriceStr(self.off_price),
                base_common.IncreaseStr(self.on_price, self.off_price),
                self.holding_days,
                sum_increase,
                capital_ratio))

    def PrintCaption(self):
        print('%-8s%-12s%-12s%-10s%-16s%-10s%-10s%-10s%-6s%-8s%-8s' % (
            'index', 
            'on_date', 
            'off_date', 
            'ts_code', 
            'pred',
            'in', 
            'out', 
            'inc', 
            'hold',
            'inc_sum',
            'capi'))
        print('-' * 100)

def CodeInPool(ts_code, pool):
    for p in pool:
        if p.ts_code == ts_code:
            return True
    return False

def FreePool(pool):
    for p in pool:
        if p.status == TS_OFF:
            return p
    return None

class DQNTest():
    # DQN Agent
    def __init__(self, dsfa, split_date, o_dl_model):
        print('DQNTest.__init__')
        self.dsfa = dsfa
        self.split_date = split_date
        self.dl_model = o_dl_model
        self.test_dataset = None
    
    def SplitDateIndex(self, dataset, train_test_split_date):
        date_col_index = self.dsfa.feature.index_date
        for iloop in range(dataset.shape[0]):
            for ts_loop in range(dataset.shape[1]):
                temp_date = dataset[iloop][ts_loop][date_col_index]
                if temp_date > 0:
                    if temp_date < train_test_split_date:
                        return iloop
                    break
        return -1

    # LoadDataset 前 dl_model 需要 LoadModel 
    def LoadDataset(self):
        dataset = self.dsfa.GetDSFa3DDataset()
        print("dataset: {}".format(dataset.shape))
        split_index = self.SplitDateIndex(dataset, self.split_date)
        if split_index == -1:
            self.test_dataset = dataset
        else:
            self.test_dataset = dataset[:split_index]
        print("test: {}".format(self.test_dataset.shape))
        self.test_features = self.test_dataset[:,:,0:self.dsfa.feature.feature_size]
        self.test_features = self.dl_model.FeaturesPretreat(self.test_features)


    def Date(self, date_index):
        for iloop in range(self.code_num):
            temp_date = self.test_dataset[date_index][iloop][self.dsfa.feature.index_date]
            if temp_date != INVALID_DATE:
                return temp_date


    def Test(self, pool_size, pred_threshold, print_trade_detail=False, show_image=False):
        if self.test_dataset == None:
            self.LoadDataset()
        date_col_index = self.dsfa.feature.index_date
        open_col_index = self.dsfa.feature.index_open
        tscode_col_index = self.dsfa.feature.index_tscode
        self.date_num = self.test_dataset.shape[0]
        self.code_num = self.test_dataset.shape[1]
        date_num = self.date_num
        code_num = self.code_num

        print("test_features:{}".format(self.test_features.shape))
        predictions = self.dl_model.Predict(self.test_features, True)
        for i in range(date_num):
            for j in range(code_num):
                if self.test_dataset[i][j][date_col_index] == 0.0:
                    predictions[i][j][0] = 0.0
        print("predictions:{}".format(predictions.shape))

        pool = []
        for iloop in range(pool_size):
            pool.append(DQNTestCodeStatus())
        
        trade_count = 0
        increase_sum = 0.0
        hold_days_sum = 0
        capital_ratio = 1.0
        capital_ratio_list = []
        increase_sum_list = []
        if print_trade_detail:
            pool[0].PrintCaption()
        for dloop in reversed(range(date_num)):  # 遍历dataset的日期
            # 更新 code_pool 内 status 非 OFF 的数据
            global_status_update_flag = False
            for p in pool:
                if p.status != TS_OFF:
                    temp_date = self.test_dataset[dloop][p.code_index][date_col_index]
                    if temp_date != INVALID_DATE:  # 未停牌
                        p.current_price = self.test_dataset[dloop][p.code_index][open_col_index]
                        p.current_pred = predictions[dloop][p.code_index][0]
                        if p.status == TS_PRE_ON:
                            p.status = TS_ON
                            p.on_date = temp_date
                            p.on_price = p.current_price
                        if p.status == TS_ON:
                            p.holding_days += 1
                            if p.current_pred < pred_threshold:
                                p.status = TS_PRE_OFF
                                p.pre_off_date = temp_date
                        elif p.status == TS_PRE_OFF:
                            p.status = TS_OFF
                            p.off_date = temp_date
                            p.off_price = p.current_price
                            p.UpdateInc()
                            # 更新全局状态
                            trade_count += 1
                            increase_sum += p.inc / pool_size
                            capital_ratio += capital_ratio / pool_size * p.inc
                            hold_days_sum += p.holding_days
                            global_status_update_flag = True
                            if print_trade_detail:
                                p.Print(trade_count, increase_sum, capital_ratio)
                            p.Reset()
            if global_status_update_flag:
                temp_date = self.Date(dloop)
                increase_sum_list.append([temp_date, increase_sum])
                capital_ratio_list.append([temp_date, capital_ratio])

            # AppendNewCode
            if FreePool(pool) != None:
                order_list = np.argsort(predictions[dloop], axis=None).tolist()[::-1]
                for c_index in order_list:
                    pred = predictions[dloop][c_index][0]
                    ts_code = self.test_dataset[dloop][c_index][tscode_col_index]
                    if pred < pred_threshold:
                        break
                    p = FreePool(pool)
                    if p == None:
                        break
                    if not CodeInPool(ts_code, pool):
                        p.status = TS_PRE_ON
                        p.ts_code = ts_code
                        p.code_index = c_index
                        p.pre_on_pred = pred
                        p.current_pred = pred
                        p.pre_on_date = self.test_dataset[dloop][c_index][date_col_index]

        if print_trade_detail:
            for p in pool:
                if p.status != TS_OFF:
                    p.Print(trade_count, increase_sum, capital_ratio)
            print('hold_days_sum: %u' % (hold_days_sum / pool_size))

        if show_image:
            np_common.Show2DData('dqn_test', [np.array(capital_ratio_list)], [], True)
            # np_common.Show2DData('dqn_test', [np.array(increase_sum_list)], [], True)
        return increase_sum, capital_ratio, trade_count








# def main(argv):
#     del argv
#     dqn_test = DQNTest()
#     dqn_test.LoadDataset(FLAGS.dataset, dqn_dataset_dataset_train_test_split_date_)
#     if FLAGS.epoch > -2:
#         dqn_test.LoadModel(FLAGS.model, FLAGS.epoch)
#         dqn_test.TestTop1(True)
#     else:
#         test_increase = []
#         for iloop in range(1000000):
#             if dqn_test.LoadModel(FLAGS.model, iloop):
#                 increase_sum, trade_count, max_Q_mean = dqn_test.TestTop1(False)
#                 test_increase.append([iloop, increase_sum])
#                 sys.stdout.write('\r%d' % (iloop))
#                 sys.stdout.flush()
#             else:
#                 break
#         if len(test_increase):
#             np.save('%s/test_increase.npy' % FLAGS.model, np.array(test_increase))
        
#     exit()

# if __name__ == "__main__":
#     flags.DEFINE_string('dataset', '-', 'dataset file name')
#     flags.DEFINE_string('model', '-', 'model path name')
#     flags.DEFINE_integer('epoch', -2, 'test model epoch, -1:model.h5, -2:test all models, >=0:model_epoch.h5')
#     app.run(main)



if __name__ == "__main__":
    pool_size = 2
    pool = []
    for iloop in range(pool_size):
        pool.append(DQNTestCodeStatus())

    pool[0].status = TS_ON
    pool[0].ts_code = 1
    pool[1].status = TS_ON
    pool[1].ts_code = 2

    p = FreePool(pool)
    if p != None:
        p.ts_code = 33
    pool[0].Print(10, 1.5, 10.0)
    pool[1].Print(12, 1.6, 12.0)
