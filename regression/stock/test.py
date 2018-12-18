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
import random

reload(sys)
sys.setdefaultencoding('utf-8')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 


model=keras.models.load_model("./model/model.h5")
mean=np.load('./model/mean.npy')
std=np.load('./model/std.npy')

print("load...")
load_data = tushare_data.GetTestData()
increase_sum=0.0
trade_count=0
feature_size = tushare_data.FeatureSize()
label_index = tushare_data.LabelColIndex()
print("%10s%10s%10s%10s%10s%10s%10s%10s%10s%10s%10s" %(
    "index", \
    "date", \
    "code", \
    "pred", \
    "t1_open_i", \
    "t1_low_i", \
    "t1_open", \
    "t1_low", \
    "buy", \
    "t5_close", \
    "act_inc"))
print("-------------------------------------------------------------------------------")
for day_loop in range(0, 30):
# features              offset = 0
#     features[5]: pre close
#     features[6]: pre close 5 avg
# + T1_close_5_avg_increse    offset = 87
# + T2_close_5_avg_increase   
# + ... 
# + T_predict_day_close 5_avg_increase 
# + T1_open_increse     offset = 92
# + T1_low_increase     offset = 93
# + T1_open             offset = 94
# + T1_low              offset = 95
# + T5_close            offset = 96
# + stock_code          offset = 97
# + T1_trade_date       offset = 99
    test_data=load_data[day_loop]
    test_features=test_data[:,0:feature_size]
    test_features=(test_features - mean) / std
    predictions = model.predict(test_features).flatten()
    predictions.shape=(len(predictions),1)
    predictions_df=pd.DataFrame(predictions, columns=['predictions'])

    current_data=test_data[:,5:7]
    current_data_df=pd.DataFrame(current_data, columns=[ \
        'pre_close', \
        'pre_close_5_avg'])

    acture_data=test_data[:,92:]
    acture_data_df=pd.DataFrame(acture_data, columns=[ \
        'T1_open_increse', \
        'T1_low_increase', \
        'T1_open', \
        'T1_low', \
        'T5_close', \
        'stock_code', \
        'T1_trade_date'])

    result = predictions_df
    result = pd.merge(result, current_data_df, left_index=True, right_index=True)
    result = pd.merge(result, acture_data_df, left_index=True, right_index=True)
    result = result.sort_values(by="predictions", ascending=False)
    result = result[:20]

    day_increase_sum=0.0
    day_trade_count=0
    day_avg_increase=0.0

    predict_trade_threshold = 15.0
    for iloop in range(0, len(result)):
        if day_trade_count < 10 :
            trade_date = result.iloc[iloop]['T1_trade_date']
            stock_code = result.iloc[iloop]['stock_code']
            pred = result.iloc[iloop]['predictions']
            pre_close = result.iloc[iloop]['pre_close']
            pre_close_5_avg = result.iloc[iloop]['pre_close_5_avg']
            t1_open_increase = result.iloc[iloop]['T1_open_increse']
            t1_low_increase = result.iloc[iloop]['T1_low_increase']
            t1_open = result.iloc[iloop]['T1_open']
            t1_low = result.iloc[iloop]['T1_low']
            t5_close = result.iloc[iloop]['T5_close']
            pred_price = pre_close_5_avg * ((pred / 100.0) + 1.0)
            pred_increase_to_pre_close = ((pred_price / pre_close) - 1.0) * 100.0
            buying_threshold = pred_increase_to_pre_close - 5.0
            if buying_threshold > 9.0 :
                buying_threshold = 9.0
            if pred > predict_trade_threshold :
                if (t1_open_increase < buying_threshold) or (t1_low_increase < buying_threshold) :
                    if (t1_open_increase < buying_threshold) :
                        buying_price = t1_open
                    else:
                        buying_price = pre_close * ((buying_threshold / 100.0) + 1.0)
                    
                    temp_increase = ((t5_close / buying_price) - 1.0) *100.0
                    day_increase_sum = day_increase_sum + temp_increase
                    day_trade_count = day_trade_count + 1
                    
                    increase_sum = increase_sum + temp_increase
                    trade_count = trade_count + 1
                
                    print("%10u%10u    %06u%10.2f%10.2f%10.2f%10.2f%10.2f%10.2f%10.2f%10.2f" %( \
                        trade_count, \
                        int(trade_date), \
                        int(stock_code), \
                        pred, \
                        t1_open_increase, \
                        t1_low_increase, \
                        t1_open, \
                        t1_low, \
                        buying_price, \
                        t5_close, \
                        temp_increase))
    if day_trade_count > 0 :
        day_avg_increase=day_increase_sum/day_trade_count
    # print("%2u day, trade_count=%u, ave_increase=%f" % (day_loop, day_trade_count, day_avg_increase))







    # for i in range(0, 100):
    #     if day_trade_count < 5 :
    #         iloop = random.randint(0, len(result))
    #         buying_threshold=0.0
    #         pred=result.iloc[iloop]['predictions']
    #         t1_open=result.iloc[iloop]['t+1_open']
    #         t1_low=result.iloc[iloop]['t+1_low']
    #         label_inc=result.iloc[iloop]['label']
    #         if (t1_open < buying_threshold) or (t1_low < buying_threshold) :
    #             if (t1_open < buying_threshold) :
    #                 buying_inc=t1_open
    #             else:
    #                 buying_inc=buying_threshold
                
    #             temp_increase=(((label_inc/100.0)+1.0)/((buying_inc/100.0)+1.0))*100.0-100.0
    #             day_increase_sum=day_increase_sum+temp_increase
    #             day_trade_count=day_trade_count+1
                
    #             increase_sum=increase_sum+temp_increase
    #             trade_count=trade_count+1
    #             print("%8u%8u%8.2f%8.2f%8.2f%8.2f%8.2f" %(trade_count, day_loop, pred, label_inc, t1_open, t1_low, temp_increase))
    # if day_trade_count > 0 :
    #     day_avg_increase=day_increase_sum/day_trade_count
    # # print("%2u day, trade_count=%u, ave_increase=%f" % (day_loop, day_trade_count, day_avg_increase))




if trade_count > 0:
    avg_increase=increase_sum/trade_count
else:
    avg_increase=0.0
print("Global, trade_count=%u, ave_increase=%f" % (trade_count, avg_increase))
