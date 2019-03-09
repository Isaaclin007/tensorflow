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

predict_data = tushare_data.GetAStockFeatures('000595.SZ', '20190110')
feature_data = predict_data[:,0:feature_size]
feature_data = (feature_data - mean) / std
predictions = model.predict(feature_data).flatten()
predictions.shape=(len(predictions),1)
predictions_df=pd.DataFrame(predictions, columns=['predictions'])

temp_index = feature_size
acture_data = predict_data[:, temp_index:temp_index+tushare_data.acture_size]
acture_data_df = pd.DataFrame(acture_data, columns=[ \
        'pre_open_increse', \
        'pre_low_increase', \
        'pre_open', \
        'pre_low', \
        'pre_close', \
        'stock_code', \
        'pre_trade_date'])

result = pd.merge(predictions_df, acture_data_df, left_index=True, right_index=True)
print(result)