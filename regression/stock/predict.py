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

model=keras.models.load_model("./model/model.h5")
mean=np.load('./model/mean.npy')
std=np.load('./model/std.npy')

print("load...")
feature_size = tushare_data.FeatureSize()
predict_data = tushare_data.GetPredictData()
feature_data = predict_data[:,0:feature_size]
feature_data = (feature_data - mean) / std
predictions = model.predict(feature_data).flatten()

stock_info = predict_data[:,feature_size:feature_size+2]
stock_info_df=pd.DataFrame(stock_info, columns=['pre_close', 'stock_code'])
predictions.shape=(len(predictions),1)
predictions_df=pd.DataFrame(predictions, columns=['predictions'])
result=pd.merge(stock_info_df, predictions_df, left_index=True, right_index=True)
result.to_csv('result.csv')
result=result.sort_values(by="predictions", ascending=False)
result.to_csv('result_sort.csv')
result=result[:20]

print("%10s%10s%10s%10s%10s" %(
    "index", \
    "code", \
    "predict", \
    "pre_close", \
    "buy_thred")
print("-------------------------------------------------------------------------------")
for iloop in range(0, len(result)):
    stock_code = result.iloc[iloop]['stock_code']
    pred = result.iloc[iloop]['predictions']
    pre_close = result.iloc[iloop]['pre_close']
    buying_threshold_increase = pred - 5.0
    if buying_threshold_increase > 9.0 :
        buying_threshold_increase = 9.0
    buying_threshold = pre_close * ((buying_threshold_increase / 100.0) + 1.0)
    print("%10u    %06u%10.2f%10.2f%10.2f%10.2f%10.2f%10.2f%10.2f%10.2f" %( \
        iloop, \
        int(stock_code), \
        pred, \
        pre_close, \
        buying_threshold))

# print(result)

# result=np.hstack((predict_stock_code, predictions))

# dtype = [('code', 'S10'), ('predict_increase', float)]
# result = np.array(result, dtype=dtype)
# print(result)