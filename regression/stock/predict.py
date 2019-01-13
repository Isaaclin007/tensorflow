# -*- coding:UTF-8 -*-

import tensorflow as tf
from tensorflow import keras
import tushare as ts
import numpy as np
import pandas as pd
import os
import time
import sys
import tushare_data

reload(sys)
sys.setdefaultencoding('utf-8')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

model=keras.models.load_model("./model/model.h5")
mean=np.load('./model/mean.npy')
std=np.load('./model/std.npy')

print("load...")
feature_size = tushare_data.feature_size
predict_data = tushare_data.GetPredictData()
feature_data = predict_data[:,0:feature_size]
feature_data = (feature_data - mean) / std
predictions = model.predict(feature_data).flatten()
predictions.shape=(len(predictions),1)
predictions_df=pd.DataFrame(predictions, columns=['predictions'])

temp_index = feature_size
acture_data = predict_data[:, temp_index:]
acture_data_df = pd.DataFrame(acture_data, columns=[ \
        'pre_open_increse', \
        'pre_low_increase', \
        'pre_open', \
        'pre_low', \
        'pre_close', \
        'stock_code', \
        'pre_trade_date'])

result = pd.merge(predictions_df, acture_data_df, left_index=True, right_index=True)
result=result.sort_values(by="predictions", ascending=False)
result.to_csv('predict_sort.csv')
result=result[:20]

print("%10s%10s%10s%10s%16s%10s" %(
    "index", \
    "code", \
    "predict", \
    "pre_close", \
    "pre_trade_date", \
    "buy_thred"))
print("-------------------------------------------------------------------------------")
for iloop in range(0, len(result)):
    stock_code = result.iloc[iloop]['stock_code']
    pred = result.iloc[iloop]['predictions']
    pre_close = result.iloc[iloop]['pre_close']
    pre_trade_date = result.iloc[iloop]['pre_trade_date']
    # pred_price = pre_close_5_avg * ((pred / 100.0) + 1.0)
    # pred_increase_to_pre_close = ((pred_price / pre_close) - 1.0) * 100.0
    # buying_threshold_increase = pred_increase_to_pre_close - 5.0
    if tushare_data.label_type == tushare_data.LABEL_PRE_CLOSE_2_TD_CLOSE:
        buying_threshold_increase = pred - 5.0
        if buying_threshold_increase > 9.0 :
            buying_threshold_increase = 9.0
        buying_threshold = pre_close * ((buying_threshold_increase / 100.0) + 1.0)
    else:
        buying_threshold = 0.0
    print("%10u    %06u%10.2f%10.2f%16s%10.2f" %( \
        iloop, \
        int(stock_code), \
        pred, \
        pre_close, \
        '%06u' % int(pre_trade_date), \
        buying_threshold))

# print(result)

# result=np.hstack((predict_stock_code, predictions))

# dtype = [('code', 'S10'), ('predict_increase', float)]
# result = np.array(result, dtype=dtype)
# print(result)