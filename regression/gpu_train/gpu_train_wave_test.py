# -*- coding:UTF-8 -*-

import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import os
import time
import sys
import random
import gpu_train as train_rnn
import gpu_train_feature as feature
import gpu_train_wave_dataset as wave_dataset

reload(sys)
sys.setdefaultencoding('utf-8')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

predict_threshold = 0

def AvgValue(sum_value, sample_num):
    if sample_num == 0:
        return 0.0
    else:
        return sum_value / sample_num

def TestEntry(test_data, print_msg, model, mean, std):
    predict_features = test_data[:, 0: feature.feature_size]
    if print_msg:
        print("predict_features: {}".format(predict_features.shape))
    col_index = feature.feature_size
    labels = test_data[:, col_index: col_index + 1]

    col_index += 1
    ts_codes = test_data[:, col_index: col_index + 1]

    col_index += 1
    on_pretrade_dates = test_data[:, col_index: col_index + 1]

    col_index += 1
    on_dates = test_data[:, col_index: col_index + 1]

    col_index += 1
    off_dates = test_data[:, col_index: col_index + 1]

    col_index += 1
    holding_days = test_data[:, col_index: col_index + 1]

    predict_features = train_rnn.FeaturesPretreat(predict_features, mean, std)
    predictions = model.predict_on_batch(predict_features)
    trade_count = 0
    increase_sum = 0.0
    holding_days_sum = 0
    max_drawdown = 0.0
    max_increase_sum = 0.0
    for iloop in range(0, len(test_data)):
        if predictions[iloop] > predict_threshold:
            if off_dates[iloop] < 20990101.0:
                increase_sum += labels[iloop]
                holding_days_sum += holding_days[iloop]
                trade_count += 1
                if max_increase_sum < increase_sum:
                    max_increase_sum = increase_sum.copy()
                temp_drawdown = max_increase_sum - increase_sum
                if max_drawdown < temp_drawdown:
                    max_drawdown = temp_drawdown.copy()
            if print_msg:
                print("%-6u%06u    %-10.0f%-10.0f%-10.0f%-10.0f%-10.2f%-10.2f%-10.2f%-10.2f%-10.2f" %( \
                    trade_count, \
                    int(ts_codes[iloop]), \
                    on_pretrade_dates[iloop], \
                    on_dates[iloop], \
                    off_dates[iloop], \
                    holding_days[iloop], \
                    predictions[iloop], \
                    labels[iloop], \
                    increase_sum, \
                    AvgValue(increase_sum, trade_count), \
                    AvgValue(increase_sum, holding_days_sum)))
    increase_score = increase_sum - (max_drawdown * 2)
    if print_msg:
        print("trade_count:%u, increase_sum:%-10.2f, max_drawdown:%.2f, score:%.1f" %( \
            trade_count, \
            increase_sum, \
            max_drawdown, \
            increase_score))
    increase_score = increase_sum - (max_drawdown * 2)
    return increase_score

if __name__ == "__main__":
    model_epoch = -1
    if len(sys.argv) > 1:
        model_epoch = int(sys.argv[1])
    dataset = wave_dataset.GetTestData()
    model, mean, std = train_rnn.LoadModel('wave', model_epoch)
    TestEntry(dataset, True, model, mean, std)


