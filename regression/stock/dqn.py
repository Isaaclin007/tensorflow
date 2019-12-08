# -*- coding:UTF-8 -*-


import tushare as ts
import numpy as np
import pandas as pd
import os
import time
import datetime
import sys
import math
import random
import tushare_data
import feature
import pp_daily_update
import dqn_dataset
from collections import deque
from tensorflow import keras
from tensorflow.python.keras import backend as K

GAMMA = 0.9 # discount factor for target Q
INITIAL_EPSILON = 0.5 # starting value of epsilon
FINAL_EPSILON = 0.01 # final value of epsilon
REPLAY_SIZE = 10000 # experience replay buffer size
BATCH_SIZE = 32 # size of minibatch
T = 10

STATUS_OFF = 0
STATUS_ON = 1
date_col_index = dqn_dataset.ACTURE_DATA_INDEX_DATE()
open_col_index = dqn_dataset.ACTURE_DATA_INDEX_OPEN()
tscode_col_index = dqn_dataset.ACTURE_DATA_INDEX_TSCODE()

def ModelFilePath():
    temp_path_name = "./model/dqn/%s" % (
                        dqn_dataset.TrainSettingName())
    return temp_path_name

def ModelFileNames():
    temp_path_name = ModelFilePath()
    model_name = "%s/model.h5" % temp_path_name
    mean_name = "%s/mean.npy" % temp_path_name
    std_name = "%s/std.npy" % temp_path_name
    return temp_path_name, model_name, mean_name, std_name

def LossTanhDiff(y_true, y_pred, e=0.1):
    # return abs(K.tanh((y_true - 5.0) * 0.4) - K.tanh((y_pred - 5.0) * 0.4))
    return abs(K.tanh(y_true * 0.2) - K.tanh(y_pred * 0.2)) # [-10, 10]

def BuildModel():
    # model
    model = keras.models.Sequential()
    model.add(keras.layers.LSTM(36, 
                                input_shape=(feature.feature_days, feature.feature_unit_size), 
                                return_sequences=False))
    model.add(keras.layers.Dense(1))

    my_optimizer = keras.optimizers.RMSprop(lr=0.004, rho=0.9, epsilon=1e-06)
    active_loss = LossTanhDiff
    model.compile(loss=active_loss, optimizer=my_optimizer, metrics=[active_loss])
    return model

def SaveModel(model, mean, std, T_index=-1):
    temp_path_name, model_name, mean_name, std_name = ModelFileNames()
    if not os.path.exists(temp_path_name):
        os.makedirs(temp_path_name)
    model.save(model_name)
    np.save(mean_name, mean)
    np.save(std_name, std)
    if T_index >= 0:
        model_name = '%s/model_%u.h5' % (temp_path_name, T_index)
        model.save(model_name)

def LoadModel():
    temp_path_name, model_name, mean_name, std_name = ModelFileNames()
    print("LoadModel: %s" % model_name)
    model = keras.models.load_model(model_name, custom_objects={'LossTanhDiff': LossTanhDiff})
    mean = np.load(mean_name)
    std = np.load(std_name)
    return model, mean, std

def ModelExist():
    # return False
    temp_path_name, model_name, mean_name, std_name = ModelFileNames()
    return (os.path.exists(model_name) and os.path.exists(mean_name) and os.path.exists(std_name))

class DQN():
    # DQN Agent
    def __init__(self):
        self.replay_buffer = deque()
        self.train_dataset, self.test_dataset = dqn_dataset.GetDataSet()
        self.step_num = 0
        self.LossClean()
        if ModelExist():
            self.epsilon = FINAL_EPSILON
            self.model, self.mean, self.std = LoadModel()
        else:
            self.epsilon = INITIAL_EPSILON
            self.model = BuildModel()
            temp_shape = self.train_dataset.shape
            train_data_2d = self.train_dataset.reshape((temp_shape[0] * temp_shape[1], temp_shape[2]))
            row_rand_array = np.arange(train_data_2d.shape[0])
            np.random.shuffle(row_rand_array)
            train_data_2d_sample = train_data_2d[row_rand_array[0:10000]]
            train_features = train_data_2d_sample[:, 0:feature.FEATURE_SIZE()]
            self.mean = train_features.mean(axis=0)
            self.std = train_features.std(axis=0)

    def FeaturesPretreat(self, input_features):
        features = (input_features - self.mean) / self.std
        if features.ndim == 1:
            features = features.reshape(1, feature.feature_days, feature.feature_unit_size)
        elif features.ndim == 2:
            features = features.reshape(features.shape[0], feature.feature_days, feature.feature_unit_size)
        elif features.ndim == 3:
            features = features.reshape(features.shape[0] * features.shape[1], feature.feature_days, feature.feature_unit_size)
        else:
            print("FeaturesPretreat.Error features.ndim:{}".format(features.ndim))
        return features

    def GetTrainActionQ(self, input_feature):
        self.epsilon -= (self.epsilon - FINAL_EPSILON) * 0.00001
        if random.random() <= self.epsilon:
            return (random.random() - 0.5)  # -0.5 ~ 0.5
        else:
            # predict
            features = self.FeaturesPretreat(input_feature)
            return self.model.predict(features)[0]

    def GetTestActionQ(self, input_feature):
        # predict
        features = self.FeaturesPretreat(input_feature)
        predict_result = self.model.predict(features)
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
            if dataset[dloop][code_index][date_col_index] != 0.0:
                return dloop
        return -1

    def PerceiveAndTrain(self, current_feature, next_feature, reward):
        self.replay_buffer.append((current_feature, next_feature, reward))
        replay_buffer_len = len(self.replay_buffer)
        if replay_buffer_len > REPLAY_SIZE:
            self.replay_buffer.popleft()

        # if replay_buffer_len > BATCH_SIZE:
        if replay_buffer_len > REPLAY_SIZE:
            self.TrainQNet()

    def TrainQNet(self):
        minibatch = random.sample(self.replay_buffer, BATCH_SIZE)
        current_feature_batch = [data[0] for data in minibatch]
        next_feature_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]

        features = self.FeaturesPretreat(next_feature_batch)
        y_batch = reward_batch + GAMMA * self.model.predict(features)
        loss = self.model.train_on_batch(features, y_batch)
        self.LossInput(loss[0])
        self.step_num += 1

    def SaveModel_(self, T_index=-1):
        SaveModel(self.model, self.mean, self.std, T_index)

    def LossInput(self, loss_value):
        self.loss_sum += loss_value
        self.loss_num += 1

    def LossMean(self):
        if self.loss_num == 0:
            return 0.0
        else:
            return self.loss_sum / self.loss_num

    def LossClean(self):
        self.loss_sum = 0.0
        self.loss_num = 0

    def Train(self):        
        date_num = self.train_dataset.shape[0]
        code_num = self.train_dataset.shape[1]
        print("date_num: %u" % date_num)
        print("code_num: %u" % code_num)
        print('\n%-8s%-8s%-8s%-12s%-12s%-12s' % ('T', 'step', 'loss', 'increase', 'trade_count', 'max_Q_mean'))
        print('-' * 60)
        for iloop in range(0, T):
            curren_status = STATUS_OFF
            for dloop in reversed(range(0, date_num)):  # 遍历train_dataset的日期
                if curren_status == STATUS_OFF:
                    code_index = random.randint(0, code_num-1)
                if self.train_dataset[dloop][code_index][date_col_index] == 0.0:
                    continue
                
                current_feature = self.train_dataset[dloop][code_index][0:feature.FEATURE_SIZE()]
                Q = self.GetTrainActionQ(current_feature)
                if Q > 0:
                    curren_status = STATUS_ON
                else:
                    curren_status = STATUS_OFF
                t1_date_index = self.NextValidDateIndex(self.train_dataset, code_index, dloop)
                if t1_date_index < 0:
                    break
                t2_date_index = self.NextValidDateIndex(self.train_dataset, code_index, t1_date_index)
                if t2_date_index < 0:
                    break
                next_feature = self.train_dataset[t1_date_index][code_index][0:feature.FEATURE_SIZE()]
                reward = self.train_dataset[t2_date_index][code_index][open_col_index] / self.train_dataset[t1_date_index][code_index][open_col_index] - 1.0
                self.PerceiveAndTrain(current_feature, next_feature, reward)
            increase, trade_count, max_Q_mean = self.TestTop1()
            print('%-8u%-8u%-8.5f%-12.3f%-12u%-12.6f' % (iloop, 
                                                         self.step_num, 
                                                         self.LossMean(), 
                                                         increase, 
                                                         trade_count, 
                                                         max_Q_mean))
            self.LossClean()
            self.SaveModel_(iloop)
                
    def Test(self):
        date_num = self.test_dataset.shape[0]
        code_num = self.test_dataset.shape[1]
        curren_status = STATUS_OFF
        trade_count = 0
        increase_sum = 0.0
        for dloop in reversed(range(0, date_num)):  # 遍历dataset的日期
            if curren_status == STATUS_OFF:
                code_index = random.randint(0, code_num-1)
            if self.test_dataset[dloop][code_index][date_col_index] == 0.0:
                continue
            
            current_feature = self.test_dataset[dloop][code_index][0:feature.FEATURE_SIZE()]
            Q = self.GetTestActionQ(current_feature)
            if Q > 0.01:
                # print(Q)
                next_status = STATUS_ON
            else:
                next_status = STATUS_OFF
            
            if curren_status == STATUS_OFF:
                if next_status == STATUS_ON:
                    curren_status = STATUS_ON
                    t1_date_index = self.NextValidDateIndex(self.test_dataset, code_index, dloop)
                    in_price = self.test_dataset[t1_date_index][code_index][open_col_index]
            else:
                if next_status == STATUS_OFF:
                    curren_status = STATUS_OFF
                    t1_date_index = self.NextValidDateIndex(self.test_dataset, code_index, dloop)
                    out_price = self.test_dataset[t1_date_index][code_index][open_col_index]
                    increase = out_price / in_price - 1.0
                    increase_sum += increase
                    trade_count += 1
        print("Test increase_sum:%.3f, trade_count:%u" % (increase_sum, trade_count))
        return increase_sum

    def TestTop1(self, print_trade_detail=False):
        test_features = self.test_dataset[:,:,0:feature.FEATURE_SIZE()]
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
            print('%-8s%-10s%-10s%-10s%-10s%-10s%-10s%-10s' % ('index', 'in_date', 'out_date', 'ts_code', 'in', 'out', 'increase', 'hold_days'))
            print('-' * 80)
        for dloop in reversed(range(0, date_num)):  # 遍历dataset的日期
            if curren_status == STATUS_OFF:
                code_index = max_Q_codes_index[dloop]
            if self.test_dataset[dloop][code_index][date_col_index] == 0.0:
                continue
            
            Q = predictions[dloop][code_index]
            if Q > 0.01:
                # print(Q)
                next_status = STATUS_ON
            else:
                next_status = STATUS_OFF
            
            if curren_status == STATUS_OFF:
                if next_status == STATUS_ON:
                    curren_status = STATUS_ON
                    t1_date_index = self.NextValidDateIndex(self.test_dataset, code_index, dloop)
                    in_price = self.test_dataset[t1_date_index][code_index][open_col_index]
            else:
                if next_status == STATUS_OFF:
                    curren_status = STATUS_OFF
                    t2_date_index = self.NextValidDateIndex(self.test_dataset, code_index, dloop)
                    out_price = self.test_dataset[t2_date_index][code_index][open_col_index]
                    increase = out_price / in_price - 1.0
                    hold_days = t1_date_index - t2_date_index
                    if print_trade_detail:
                        print('%-8u%-10.0f%-10.0f%-10s%-10.2f%-10.2f%-10.4f%-10u' % (trade_count, 
                                self.test_dataset[t1_date_index][code_index][date_col_index], 
                                self.test_dataset[t2_date_index][code_index][date_col_index], 
                                '%06u' % self.test_dataset[t1_date_index][code_index][tscode_col_index],
                                in_price,
                                out_price,
                                increase,
                                hold_days))
                    increase_sum += increase
                    trade_count += 1
                    hold_days_sum += hold_days
        if print_trade_detail:
            print('%-8s%-10s%-10s%-10s%-10s%-10s%-10.4f%-10u' % ('sum', '--', '--', '--', '--', '--', increase_sum, hold_days_sum))
        return increase_sum, trade_count, max_Q_mean





if __name__ == "__main__":
    dqn = DQN()
    dqn.Train()
    # dqn.SaveModel_()
    # dqn.TestTop1(True)
    # dqn.Test()


