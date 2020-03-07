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

    def NextValidDateIndex(self, dataset, code_index, current_date_index):
        for dloop in reversed(range(0, current_date_index)):
            if dataset[dloop][code_index][self.dsfa.feature.index_date] != 0.0:
                return dloop
        return -1

    def Test(self, pred_threshold, print_trade_detail=False, show_image=False):
        if self.test_dataset == None:
            self.LoadDataset()
        date_col_index = self.dsfa.feature.index_date
        open_col_index = self.dsfa.feature.index_open
        tscode_col_index = self.dsfa.feature.index_tscode
        date_num = self.test_dataset.shape[0]
        code_num = self.test_dataset.shape[1]

        print("test_features:{}".format(self.test_features.shape))
        predictions = self.dl_model.Predict(self.test_features, True)
        for i in range(date_num):
            for j in range(code_num):
                if self.test_dataset[i][j][date_col_index] == 0.0:
                    predictions[i][j][0] = 0.0
        print("predictions:{}".format(predictions.shape))
        max_Q_codes_index = np.argmax(predictions, axis=1).flatten()
        print("max_Q_codes_index:{}".format(max_Q_codes_index.shape))
        max_Q_codes_value = np.amax(predictions, axis=1)
        max_Q_mean = np.mean(max_Q_codes_value)
        # print("max_Q_mean:{}".format(max_Q_mean))
        
        curren_status = TS_OFF
        trade_count = 0
        increase_sum = 0.0
        hold_days_sum = 0
        capital_ratio = 1.0
        capital_ratio_list = []
        increase_sum_list = []
        if print_trade_detail:
            print('%-8s%-10s%-10s%-10s%-10s%-10s%-10s%-10s%-10s' % ('index', 'in_date', 'out_date', 'ts_code', 'pred','in', 'out', 'increase', 'hold_days'))
            print('-' * 80)
        dloop = date_num - 1
        while dloop >= 0:  # 遍历dataset的日期
            if curren_status == TS_OFF:
                code_index = max_Q_codes_index[dloop]
            if self.test_dataset[dloop][code_index][date_col_index] == 0.0:
                dloop -= 1
                continue
            
            Q = predictions[dloop][code_index]
            if Q >= pred_threshold:
                # print(Q)
                next_status = TS_ON
            else:
                next_status = TS_OFF
            
            # print('%u, %u' % (dloop, next_status))

            if curren_status == TS_OFF:
                if next_status == TS_ON:
                    curren_status = TS_ON
                    t1_date_index = self.NextValidDateIndex(self.test_dataset, code_index, dloop)
                    if t1_date_index < 0:
                        break
                    in_price = self.test_dataset[t1_date_index][code_index][open_col_index]
                    in_pred = Q
                    dloop = t1_date_index
                else:
                    dloop -= 1
            else:
                if next_status == TS_OFF:
                    curren_status = TS_OFF
                    t2_date_index = self.NextValidDateIndex(self.test_dataset, code_index, dloop)
                    if t2_date_index < 0:
                        break
                    out_price = self.test_dataset[t2_date_index][code_index][open_col_index]
                    increase = out_price / in_price - 1.0
                    hold_days = t1_date_index - t2_date_index
                    increase_sum += increase
                    capital_ratio *= (increase + 1.0)
                    trade_count += 1
                    hold_days_sum += hold_days
                    dloop = t2_date_index
                    if print_trade_detail:
                        print('%-8u%-10.0f%-10.0f%-10s%-10.2f%-10.2f%-10.2f%-10.4f%-10u%-10.4f' % (trade_count, 
                                self.test_dataset[t1_date_index][code_index][date_col_index], 
                                self.test_dataset[t2_date_index][code_index][date_col_index], 
                                '%06u' % self.test_dataset[t1_date_index][code_index][tscode_col_index],
                                in_pred,
                                in_price,
                                out_price,
                                increase,
                                hold_days,
                                capital_ratio))
                    temp_date = self.test_dataset[t2_date_index][code_index][date_col_index]
                    capital_ratio_list.append([temp_date, capital_ratio])
                    increase_sum_list.append([temp_date, increase_sum])
                else:
                    dloop -= 1
        if print_trade_detail:
            print('%-8s%-10s%-10s%-10s%-10s%-10s%-10s%-10.4f%-10u%-10.4f' % ('sum', '--', '--', '--', '--', '--', '--', increase_sum, hold_days_sum, capital_ratio))

        if show_image:
            # np_common.Show2DData('dqn_test', [np.array(capital_ratio_list)], [], True)
            np_common.Show2DData('dqn_test', [np.array(increase_sum_list)], [], True)
        return increase_sum, trade_count, max_Q_mean

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



