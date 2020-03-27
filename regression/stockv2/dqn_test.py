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

class MaxDrawdown():
    def __init__(self):
        self.ResetMaxDrawdown()

    def ResetMaxDrawdown(self):
        self.capital_ratio_max_value = 1.0
        self.capital_ratio_max_drawdown = 0.0

    def UpdateMaxDrawdown(self, capital_ratio):
        if capital_ratio > self.capital_ratio_max_value:
            self.capital_ratio_max_value = capital_ratio
        drawdown = 1.0 - (capital_ratio / self.capital_ratio_max_value)
        if drawdown > self.capital_ratio_max_drawdown:
            self.capital_ratio_max_drawdown = drawdown

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
        self.on_open_inc = 0.0
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
        print('%-8u%-12s%-12s%-10s%-8.2f%-8.2f%-8.2f%-10s%-10s%-10s%-6u%-8.2f%-8.2E' % (
                trade_index, 
                base_common.TradeDateStr(self.pre_on_date, self.on_date),
                base_common.TradeDateStr(self.pre_off_date, self.off_date),
                base_common.CodeStr(self.ts_code),
                self.pre_on_pred,
                self.current_pred,
                self.on_open_inc,
                base_common.PriceStr(self.on_price),
                base_common.PriceStr(self.off_price),
                base_common.IncreaseStr(self.on_price, self.off_price),
                self.holding_days,
                sum_increase,
                capital_ratio))

    def PrintCaption(self):
        print('%-8s%-12s%-12s%-10s%-8s%-8s%-8s%-10s%-10s%-10s%-6s%-8s%-8s' % (
            'index', 
            'on_date', 
            'off_date', 
            'ts_code', 
            'p_pred',
            'c_pred',
            'oo_inc',
            'on', 
            'off', 
            'inc', 
            'hold',
            'inc_sum',
            'capi'))
        print('-' * 120)

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

def NextOpenIncrease(acture_data, date_index, code_index):
    temp_index = date_index - 1
    while temp_index > 0:
        if INVALID_DATE != acture_data[temp_index][code_index][feature.ADI_DATE]:
            return acture_data[temp_index][code_index][feature.ADI_OPEN_INCREASE]
        temp_index -= 1
    return 0.0

def Date(acture_data, date_index):
    for iloop in range(acture_data.shape[1]):
        temp_date = acture_data[date_index][iloop][feature.ADI_DATE]
        if temp_date != INVALID_DATE:
            return temp_date

# acture_data: 3D data
# predictions: 2D data
def TestLowLevel(pool_size, acture_data, predictions, pred_threshold, print_trade_detail=False, show_image=False):
    date_num = acture_data.shape[0]
    code_num = acture_data.shape[1]
    
    predictions[acture_data[:,:,feature.ADI_DATE] == INVALID_DATE] = 0.0
    if print_trade_detail:
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
    MDD = MaxDrawdown()
    pos_num = 0
    pos_sum = 0.0
    neg_num = 0
    neg_sum = 0.0
    if print_trade_detail:
        pool[0].PrintCaption()
    for dloop in reversed(range(date_num)):  # 遍历dataset的日期
        # 更新 code_pool 内 status 非 OFF 的数据
        global_status_update_flag = False
        for p in pool:
            if p.status != TS_OFF:
                temp_date = acture_data[dloop][p.code_index][feature.ADI_DATE]
                if temp_date != INVALID_DATE:  # 未停牌
                    p.current_price = acture_data[dloop][p.code_index][feature.ADI_OPEN]
                    p.current_pred = predictions[dloop][p.code_index]
                    if p.status == TS_PRE_ON:
                        p.status = TS_ON
                        p.on_date = temp_date
                        p.on_price = p.current_price
                        p.on_open_inc = acture_data[dloop][p.code_index][feature.ADI_OPEN_INCREASE]
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
                        if p.inc > 0:
                            pos_num += 1
                            pos_sum += p.inc
                        else:
                            neg_num += 1
                            neg_sum += p.inc

                        # 更新全局状态
                        trade_count += 1
                        increase_sum += p.inc / pool_size
                        capital_ratio += capital_ratio / pool_size * p.inc
                        MDD.UpdateMaxDrawdown(capital_ratio)
                        hold_days_sum += p.holding_days
                        global_status_update_flag = True
                        if print_trade_detail:
                            p.Print(trade_count, increase_sum, capital_ratio)
                        p.Reset()
        if global_status_update_flag:
            temp_date = Date(acture_data, dloop)
            increase_sum_list.append([temp_date, increase_sum])
            capital_ratio_list.append([temp_date, capital_ratio])

        # AppendNewCode
        if FreePool(pool) != None:
            order_list = np.argsort(predictions[dloop], axis=None).tolist()[::-1]
            for c_index in order_list:
                pred = predictions[dloop][c_index]
                ts_code = acture_data[dloop][c_index][feature.ADI_TSCODE]
                if pred < pred_threshold:
                    break
                # if acture_data[dloop][c_index][feature.ADI_CLOSE] < \
                #    acture_data[dloop][c_index][feature.ADI_CLOSE_100_AVG]:
                #     continue
                if (acture_data[dloop][c_index][feature.ADI_VOL] < 10000) or\
                   (acture_data[dloop][c_index][feature.ADI_VOL] < acture_data[dloop][c_index][feature.ADI_VOL_100_AVG] * 0.2):
                p = FreePool(pool)
                if p == None:
                    break
                open_inc = NextOpenIncrease(acture_data, dloop, c_index)
                # if (not CodeInPool(ts_code, pool)) and (open_inc > 9.5):
                if not CodeInPool(ts_code, pool):
                    p.status = TS_PRE_ON
                    p.ts_code = ts_code
                    p.code_index = c_index
                    p.pre_on_pred = pred
                    p.current_pred = pred
                    p.pre_on_date = acture_data[dloop][c_index][feature.ADI_DATE]

    if print_trade_detail:
        for p in pool:
            if p.status != TS_OFF:
                p.Print(trade_count, increase_sum, capital_ratio)
        print('hold_days_sum: %u' % (hold_days_sum / pool_size))
        print('capital_ratio_max_drawdown: %.2f' % MDD.capital_ratio_max_drawdown)
        pos_avg = pos_sum / pos_num if pos_num > 0 else 0.0
        neg_avg = neg_sum / neg_num if neg_num > 0 else 0.0
        print('pos: %6.2f, %4u, %6.2f' % (pos_sum, pos_num, pos_avg))
        print('neg: %6.2f, %4u, %6.2f' % (neg_sum, neg_num, neg_avg))

        dloop = 0
        order_list = np.argsort(predictions[dloop], axis=None).tolist()[::-1]
        cnt = 0
        for c_index in order_list:
            pred = predictions[dloop][c_index]
            ts_code = acture_data[dloop][c_index][feature.ADI_TSCODE]
            date = acture_data[dloop][c_index][feature.ADI_DATE]
            if date != INVALID_DATE:
                print('%-8s%-12s%-12s%-10.2f' % (
                    '--', 
                    '%u+1' % int(date),
                    '%06u' % int(ts_code), 
                    pred))
                cnt += 1
                if cnt > 10:
                    break

    if show_image:
        np_common.Show2DData('dqn_test', [np.array(capital_ratio_list)], [], True)
        # np_common.Show2DData('dqn_test', [np.array(increase_sum_list)], [], True)
    return increase_sum,\
            capital_ratio,\
            trade_count,\
            int((hold_days_sum / pool_size)),\
            MDD.capital_ratio_max_drawdown

class DQNTest():
    # DQN Agent
    def __init__(self, dsfa, split_date, o_dl_model):
        print('DQNTest.__init__')
        self.dsfa = dsfa
        self.split_date = split_date
        self.dl_model = o_dl_model
        self.test_dataset = []
    
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
        print("test_features:{}".format(self.test_features.shape))
        self.acture_data = self.test_dataset[:,:,self.dsfa.feature.feature_size:]
        print("acture_data:{}".format(self.acture_data.shape))

    def ShowAvgPred(self, predictions):
        avg_pred_list = []
        for iloop in range(self.date_num):
            valid_pos = self.test_dataset[iloop,:,self.dsfa.feature.index_date] > 0
            valid_pred = predictions[iloop,:,0][valid_pos]
            temp_date = self.Date(iloop)
            avg_pred = np.mean(valid_pred)
            avg_pred_list.append([temp_date, avg_pred])
        if show_image:
            np_common.Show2DData('dqn_test_avg_pred', [np.array(avg_pred_list)], [], True)

    def AvgPredList(self, predictions):
        avg_pred_list = []
        for iloop in range(self.date_num):
            valid_pos = self.test_dataset[iloop,:,self.dsfa.feature.index_date] > 0
            valid_pred = predictions[iloop,:,0][valid_pos]
            avg_pred = np.mean(valid_pred)
            avg_pred_list.append(avg_pred)
        return avg_pred_list

    def Test(self, pool_size, pred_threshold, print_trade_detail=False, show_image=False):
        if len(self.test_dataset) == 0:
            self.LoadDataset()
        predictions = self.dl_model.Predict(self.test_features, True)
        predictions = predictions.reshape(predictions.shape[0:2])
        return TestLowLevel(pool_size, self.acture_data, predictions, pred_threshold, print_trade_detail, show_image)


    def TestAllModels(self, pool_size, pred_threshold):
        if len(self.test_dataset) == 0:
            self.LoadDataset()
        print('%-10s%-10s%-10s%-10s%-10s%-10s' % ('epoch', 'sum_inc', 'capital', 'capi_MDD', 'trd_cnt', 'h_days'))
        print('-' * 60)
        for epoch in range(1, self.dl_model.MaxModelEpoch() + 1):
            if self.dl_model.ModelExist(epoch):
                self.dl_model.LoadModel(epoch)
                sum_inc,capital,trd_cnt,h_days,capi_MDD = self.Test(pool_size, pred_threshold, False, False)
                print('%-10u%-10.2f%-10.2E%-10.2f%-10u%-10u' % (epoch,
                                                            sum_inc, 
                                                            capital, 
                                                            capi_MDD, 
                                                            trd_cnt, 
                                                            h_days))






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
