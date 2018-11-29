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
load_data=np.load("./temp_data/test_data.npy")
print("load_data: {}".format(load_data.shape))
increase_sum=0.0
trade_count=0
print("%8s%8s%8s%8s%8s%8s%8s" %("index", "day", "pred", "label", "t1_open", "t1_low", "act_inc"))
print("---------------------------------------")
for day_loop in range(0, load_data.shape[0]):
    test_data=load_data[day_loop]
    feature_size=tushare_data.FeatureSize()
    test_features=test_data[:,0:feature_size]
    test_features=(test_features - mean) / std
    predictions = model.predict(test_features).flatten()
    predictions.shape=(len(predictions),1)
    acture_data=test_data[:,feature_size:]
    label_index=tushare_data.LabelColIndex()
    label_data=test_data[:,label_index:label_index+1]
    predictions_df=pd.DataFrame(predictions, columns=['predictions'])
    label_df=pd.DataFrame(label_data, columns=['label'])
    acture_data_df=pd.DataFrame(acture_data, columns=['t+1_inc', 't+2_inc', 't+3_inc', 't+4_inc', 't+5_inc', 't+1_open', 't+1_low'])
    result_temp=pd.merge(predictions_df, label_df, left_index=True, right_index=True)
    result=pd.merge(result_temp, acture_data_df, left_index=True, right_index=True)

    result=result.sort_values(by="predictions", ascending=False)
    # result.to_csv('result_sort.csv')
    # print("\n\n%2u day result:" % day_loop)
    result=result[:20]
    # print(result)

    day_increase_sum=0.0
    day_trade_count=0
    day_avg_increase=0.0
    # print("\n\n%2u day trade:" % day_loop)
    
    predict_trade_threshold=10.0
    buying_threshold=7.0
    for iloop in range(0, len(result)):
        if day_trade_count < 10 :
            pred=result.iloc[iloop]['predictions']
            t1_open=result.iloc[iloop]['t+1_open']
            t1_low=result.iloc[iloop]['t+1_low']
            label_inc=result.iloc[iloop]['label']
            if pred > predict_trade_threshold :
                if (t1_open < buying_threshold) or (t1_low < buying_threshold) :
                    if (t1_open < buying_threshold) :
                        buying_inc=t1_open
                    else:
                        buying_inc=buying_threshold
                    day_trade_count=day_trade_count+1
                    temp_increase=(((label_inc/100.0)+1.0)/((buying_inc/100.0)+1.0))*100.0-100.0
                    day_increase_sum=day_increase_sum+temp_increase
                    trade_count=trade_count+1
                    increase_sum=increase_sum+temp_increase
                    print("%8u%8u%8.2f%8.2f%8.2f%8.2f%8.2f" %(trade_count, day_loop, pred, label_inc, t1_open, t1_low, temp_increase))
    if day_trade_count > 0 :
        day_avg_increase=day_increase_sum/day_trade_count
    # print("%2u day, trade_count=%u, ave_increase=%f" % (day_loop, day_trade_count, day_avg_increase))
avg_increase=increase_sum/trade_count
print("Global, trade_count=%u, ave_increase=%f" % (trade_count, avg_increase))
