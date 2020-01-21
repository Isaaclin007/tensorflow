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
from tensorflow import keras
from tensorflow.python.keras import backend as K
import loss

FLAGS = flags.FLAGS

import_feature = os.path.exists('feature.py')
if import_feature:
    import feature

import_dqn_dataset = os.path.exists('dqn_dataset.py')
if import_dqn_dataset:
    import dqn_dataset

if import_dqn_dataset and import_feature:
    dqn_dataset_ACTURE_DATA_INDEX_DATE_ = dqn_dataset.ACTURE_DATA_INDEX_DATE()
    dqn_dataset_ACTURE_DATA_INDEX_OPEN_ = dqn_dataset.ACTURE_DATA_INDEX_OPEN()
    dqn_dataset_ACTURE_DATA_INDEX_TSCODE_ = dqn_dataset.ACTURE_DATA_INDEX_TSCODE()
    feature_FEATURE_SIZE_ = feature.FEATURE_SIZE()
    feature_feature_days_ = feature.feature_days
    feature_feature_unit_size_ = feature.feature_unit_size
    dqn_dataset_dataset_train_test_split_date_ = dqn_dataset.dataset_train_test_split_date
else:
    dqn_dataset_ACTURE_DATA_INDEX_DATE_ = 156
    dqn_dataset_ACTURE_DATA_INDEX_OPEN_ = 152
    dqn_dataset_ACTURE_DATA_INDEX_TSCODE_ = 155
    feature_FEATURE_SIZE_ = 150
    feature_feature_days_ = 30
    feature_feature_unit_size_ = 5
    dqn_dataset_dataset_train_test_split_date_ = 20170101

STATUS_OFF = 0
STATUS_ON = 1
STATUS_PRE_OFF = 2
STATUS_PRE_ON = 3

class DQNTest():
    # DQN Agent
    def __init__(self):
        print('DQNTest.__init__')
    
    def LoadModel(self, model_path, epoch=-1):
        if epoch == -1:
            model_name = "%s/model.h5" % model_path
        else:
            model_name = "%s/model_%u.h5" % (model_path, epoch)
        mean_name = "%s/mean.npy" % model_path
        std_name = "%s/std.npy" % model_path
        if (os.path.exists(model_name) and os.path.exists(mean_name) and os.path.exists(std_name)):
            self.model = keras.models.load_model(model_name, custom_objects=loss.LossDict())
            self.mean = np.load(mean_name)
            self.std = np.load(std_name)
            return True
        else:
            return False

    def SetModel(self, model, mean, std):
        self.model = model
        self.mean = mean
        self.std = std
    
    def SplitDateIndex(self, dataset, train_test_split_date):
        for iloop in range(dataset.shape[0]):
            for ts_loop in range(dataset.shape[1]):
                temp_date = dataset[iloop][ts_loop][dqn_dataset_ACTURE_DATA_INDEX_DATE_]
                if temp_date > 0:
                    if temp_date < train_test_split_date:
                        return iloop
                    break
        return -1

    def LoadDataset(self, dataset_file_name, train_test_split_date, dataset_option='test'):
        dataset = np.load(dataset_file_name)
        print("dataset: {}".format(dataset.shape))
        split_index = self.SplitDateIndex(dataset, train_test_split_date)
        if dataset_option == 'train':
            self.test_dataset = dataset[split_index:]
        elif dataset_option == 'test':
            self.test_dataset = dataset[:split_index]
        print("test: {}".format(self.test_dataset.shape))

    def FeaturesPretreat(self, input_features):
        features = (input_features - self.mean) / self.std
        if features.ndim == 1:
            features = features.reshape(1, feature_feature_days_, feature_feature_unit_size_)
        elif features.ndim == 2:
            features = features.reshape(features.shape[0], feature_feature_days_, feature_feature_unit_size_)
        elif features.ndim == 3:
            features = features.reshape(features.shape[0] * features.shape[1], feature_feature_days_, feature_feature_unit_size_)
        else:
            print("FeaturesPretreat.Error features.ndim:{}".format(features.ndim))
        return features

    def GetTestActionQ(self, input_feature):
        # predict
        features = self.FeaturesPretreat(input_feature)
        predict_result = self.model.predict(features, batch_size=10240)
        if features.ndim == 1:
            return predict_result[0]
        elif features.ndim == 2:
            return predict_result
        elif features.ndim == 3:
            return predict_result.reshape(input_feature.shape[0], input_feature.shape[1])
        else:
            print("GetTestActionQ.Error features.ndim:{}".format(features.ndim))


    def NextValidDateIndex(self, dataset, code_index, current_date_index):
        for dloop in reversed(range(0, current_date_index)):
            if dataset[dloop][code_index][dqn_dataset_ACTURE_DATA_INDEX_DATE_] != 0.0:
                return dloop
        return -1

    def TestTop1(self, print_trade_detail=False):
        date_col_index = dqn_dataset_ACTURE_DATA_INDEX_DATE_
        open_col_index = dqn_dataset_ACTURE_DATA_INDEX_OPEN_
        tscode_col_index = dqn_dataset_ACTURE_DATA_INDEX_TSCODE_

        test_features = self.test_dataset[:,:,0:feature_FEATURE_SIZE_]
        predictions = self.GetTestActionQ(test_features)
        # print("predictions:{}".format(predictions.shape))
        max_Q_codes_index = np.argmax(predictions, axis=1)
        # print("max_Q_codes_index:{}".format(max_Q_codes_index.shape))
        max_Q_codes_value = np.amax(predictions, axis=1)
        max_Q_mean = np.mean(max_Q_codes_value)
        # print("max_Q_mean:{}".format(max_Q_mean))
        
        date_num = self.test_dataset.shape[0]
        code_num = self.test_dataset.shape[1]
        curren_status = STATUS_OFF
        trade_count = 0
        increase_sum = 0.0
        hold_days_sum = 0
        if print_trade_detail:
            print('%-8s%-10s%-10s%-10s%-10s%-10s%-10s%-10s%-10s' % ('index', 'in_date', 'out_date', 'ts_code', 'pred','in', 'out', 'increase', 'hold_days'))
            print('-' * 80)
        dloop = date_num - 1
        while dloop >= 0:  # 遍历dataset的日期
            if curren_status == STATUS_OFF:
                code_index = max_Q_codes_index[dloop]
            if self.test_dataset[dloop][code_index][date_col_index] == 0.0:
                dloop -= 1
                continue
            
            Q = predictions[dloop][code_index]
            if Q > 0:
                # print(Q)
                next_status = STATUS_ON
            else:
                next_status = STATUS_OFF
            
            if curren_status == STATUS_OFF:
                if next_status == STATUS_ON:
                    curren_status = STATUS_ON
                    t1_date_index = self.NextValidDateIndex(self.test_dataset, code_index, dloop)
                    if t1_date_index < 0:
                        break
                    in_price = self.test_dataset[t1_date_index][code_index][open_col_index]
                    in_pred = Q
                    dloop = t1_date_index
                else:
                    dloop -= 1
            else:
                if next_status == STATUS_OFF:
                    curren_status = STATUS_OFF
                    t2_date_index = self.NextValidDateIndex(self.test_dataset, code_index, dloop)
                    if t2_date_index < 0:
                        break
                    out_price = self.test_dataset[t2_date_index][code_index][open_col_index]
                    increase = out_price / in_price - 1.0
                    hold_days = t1_date_index - t2_date_index
                    if print_trade_detail:
                        print('%-8u%-10.0f%-10.0f%-10s%-10.2f%-10.2f%-10.2f%-10.4f%-10u' % (trade_count, 
                                self.test_dataset[t1_date_index][code_index][date_col_index], 
                                self.test_dataset[t2_date_index][code_index][date_col_index], 
                                '%06u' % self.test_dataset[t1_date_index][code_index][tscode_col_index],
                                in_pred,
                                in_price,
                                out_price,
                                increase,
                                hold_days))
                    increase_sum += increase
                    trade_count += 1
                    hold_days_sum += hold_days
                    dloop = t2_date_index
                else:
                    dloop -= 1
        if print_trade_detail:
            print('%-8s%-10s%-10s%-10s%-10s%-10s%-10s%-10.4f%-10u' % ('sum', '--', '--', '--', '--', '--', '--', increase_sum, hold_days_sum))
        return increase_sum, trade_count, max_Q_mean




def main(argv):
    del argv
    dqn_test = DQNTest()
    dqn_test.LoadDataset(FLAGS.dataset, dqn_dataset_dataset_train_test_split_date_)
    if FLAGS.epoch > -2:
        dqn_test.LoadModel(FLAGS.model, FLAGS.epoch)
        dqn_test.TestTop1(True)
    else:
        test_increase = []
        for iloop in range(1000000):
            if dqn_test.LoadModel(FLAGS.model, iloop):
                increase_sum, trade_count, max_Q_mean = dqn_test.TestTop1(False)
                test_increase.append([iloop, increase_sum])
                sys.stdout.write('\r%d' % (iloop))
                sys.stdout.flush()
            else:
                break
        if len(test_increase):
            np.save('%s/test_increase.npy' % FLAGS.model, np.array(test_increase))
        
    exit()

if __name__ == "__main__":
    flags.DEFINE_string('dataset', '-', 'dataset file name')
    flags.DEFINE_string('model', '-', 'model path name')
    flags.DEFINE_integer('epoch', -2, 'test model epoch, -1:model.h5, -2:test all models, >=0:model_epoch.h5')
    app.run(main)



