# -*- coding:UTF-8 -*-

import tensorflow as tf
from tensorflow import keras
import tushare as ts
import numpy as np
import pandas as pd
import os
import time
import sys

reload(sys)
sys.setdefaultencoding('utf-8')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

model=keras.models.load_model("model.h5")
# optimizer = tf.train.RMSPropOptimizer(0.001)
# model.compile(loss='mse',
#                 optimizer=optimizer,
#                 metrics=['mae'])
mean=np.load('mean.npy')
std=np.load('std.npy')

print("load...")
test_data=np.load("day_test_data.npy")
print("test_data: {}".format(test_data.shape))
col_num=test_data.shape[1]
test_features=test_data[:,0:col_num-2]
test_features=(test_features - mean) / std
predictions = model.predict(test_features).flatten()
predictions.shape=(len(predictions),1)
increase=test_data[:,col_num-2:]
predictions_df=pd.DataFrame(predictions, columns=['predictions'])
increase_df=pd.DataFrame(increase, columns=['increase1', 'increase2'])
result=pd.merge(increase_df, predictions_df, left_index=True, right_index=True)

result=result.sort_values(by="predictions", ascending=False)
result.to_csv('result_sort.csv')
result=result[:50]
print(result)

increase_sum=0.0
trade_count=0
for iloop in range(0, len(result)):
    if result['predictions'] > 5.0 :
        if result['increase1'] < 9.0 :
            increase_sum+=increase2
            trade_count=trade_count+1
avg_increase=increase_sum/trade_count
print("avg_increase: %f" % avg_increase)