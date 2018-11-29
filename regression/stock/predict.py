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
predict_data=np.load("./temp_data/predict_data.npy")
predict_data = (predict_data - mean) / std
predict_stock_code=np.load("./temp_data/predict_stock_code.npy")
predictions = model.predict(predict_data).flatten()

predict_stock_code.shape=(len(predict_stock_code),1)
predictions.shape=(len(predictions),1)
print("predict_stock_code: {}".format(predict_stock_code.shape))
print("predictions: {}".format(predictions.shape))
predict_stock_code_df=pd.DataFrame(predict_stock_code, columns=['stock_code'])
predictions_df=pd.DataFrame(predictions, columns=['predictions'])
result=pd.merge(predict_stock_code_df, predictions_df, left_index=True, right_index=True)
result.to_csv('result.csv')
result=result.sort_values(by="predictions", ascending=False)
result.to_csv('result_sort.csv')
result=result[:20]
print(result)

# result=np.hstack((predict_stock_code, predictions))

# dtype = [('code', 'S10'), ('predict_increase', float)]
# result = np.array(result, dtype=dtype)
# print(result)