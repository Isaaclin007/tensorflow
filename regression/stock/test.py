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
acture_size = tushare_data.ActureSize()
print("%10s%10s%10s%10s%10s%10s%10s%10s%10s%10s%10s%10s%10s" %(
    "index", \
    "indate", \
    "outdate", \
    "code", \
    "pred", \
    "t0_open_i", \
    "t0_low_i", \
    "t0_open", \
    "t0_low", \
    "buy", \
    "out", \
    "td_close", \
    "act_inc"))
print("-------------------------------------------------------------------------------")
for day_loop in range(0, (tushare_data.test_day_count / tushare_data.test_day_sample)):
    test_data=load_data[day_loop]

    temp_index = tushare_data.TestDataLastPredictFeatureOffset()
    predict_features = test_data[:, temp_index: temp_index + feature_size]
    predict_features = (predict_features - mean) / std
    predictions = model.predict(predict_features).flatten()
    predictions.shape = (len(predictions),1)
    predictions_df = pd.DataFrame(predictions, columns=['pred'])
    
    if tushare_data.test_acture_data_with_feature:
        for iloop in range(0, tushare_data.predict_day_count):
            temp_index = tushare_data.TestDataMonitorFeatureOffset(iloop)
            monitor_features = test_data[:, temp_index: temp_index + feature_size]
            monitor_features = (monitor_features - mean) / std
            monitor_predictions = model.predict(monitor_features).flatten()
            monitor_predictions.shape = (len(monitor_predictions), 1)
            temp_caption = 'mpred_%d' % iloop
            monitor_predictions_df = pd.DataFrame(monitor_predictions, columns=[temp_caption])
            predictions_df = pd.merge(predictions_df, monitor_predictions_df, left_index=True, right_index=True)

    temp_index = tushare_data.TestDataLastPredictActureOffset() + tushare_data.ACTURE_DATA_INDEX_CLOSE
    current_data=test_data[:, temp_index: temp_index+1]
    current_data_df=pd.DataFrame(current_data, columns=['pre_close'])

    temp_index = tushare_data.TestDataMonitorActureOffset(0)
    t0_acture_data = test_data[:, temp_index: temp_index+acture_size]
    acture_data_df = pd.DataFrame(t0_acture_data, columns=[ \
        'T0_open_increse', \
        'T0_low_increase', \
        'T0_open', \
        'T0_low', \
        'T0_close', \
        'stock_code', \
        'T0_trade_date'])
    
    for iloop in range(1, tushare_data.predict_day_count):
        temp_acture_index = tushare_data.TestDataMonitorActureOffset(iloop)
        temp_index = temp_acture_index + tushare_data.ACTURE_DATA_INDEX_OPEN
        tn_acture_data = test_data[:, temp_index:temp_index+1]
        temp_caption = 'T%d_open' % (iloop)
        temp_df = pd.DataFrame(tn_acture_data, columns=[temp_caption])
        acture_data_df = pd.merge(acture_data_df, temp_df, left_index=True, right_index=True)

        temp_index = temp_acture_index + tushare_data.ACTURE_DATA_INDEX_CLOSE
        tn_acture_data = test_data[:, temp_index:temp_index+1]
        temp_caption = 'T%d_close' % (iloop)
        temp_df = pd.DataFrame(tn_acture_data, columns=[temp_caption])
        acture_data_df = pd.merge(acture_data_df, temp_df, left_index=True, right_index=True)
        
    temp_index = tushare_data.TestDataLastMonitorActureOffset()
    td_acture_data = test_data[:, temp_index: temp_index+acture_size]
    temp_df=pd.DataFrame(td_acture_data, columns=[ \
        'Td_open_increse', \
        'Td_low_increase', \
        'Td_open', \
        'Td_low', \
        'Td_close', \
        'Td_stock_code', \
        'Td_trade_date'])
    acture_data_df = pd.merge(acture_data_df, temp_df, left_index=True, right_index=True)

    result = predictions_df
    result = pd.merge(result, current_data_df, left_index=True, right_index=True)
    result = pd.merge(result, acture_data_df, left_index=True, right_index=True)
    result = result.sort_values(by = 'pred', ascending=False)
    # result.to_csv(('./result_%d_sort.csv' % day_loop))
    result = result[:20]
    if day_loop == 0:
        result_sum = result
    else:
        result_sum = result_sum.append(result)

    day_increase_sum=0.0
    day_trade_count=0
    day_avg_increase=0.0

    predict_trade_threshold = 9.0
    for iloop in range(0, len(result)):
        if day_trade_count < 10 :
            in_trade_date = result.iloc[iloop]['T0_trade_date']
            out_trade_date = result.iloc[iloop]['Td_trade_date']
            stock_code = result.iloc[iloop]['stock_code']
            pred = result.iloc[iloop]['pred']
            pre_close = result.iloc[iloop]['pre_close']
            t0_open_increase = result.iloc[iloop]['T0_open_increse']
            t0_low_increase = result.iloc[iloop]['T0_low_increase']
            t0_open = result.iloc[iloop]['T0_open']
            t0_low = result.iloc[iloop]['T0_low']
            td_close = result.iloc[iloop]['Td_close']
            buying_threshold = pred - 5.0
            if buying_threshold > 9.0 :
                buying_threshold = 9.0
            if pred > predict_trade_threshold :
                if (t0_open_increase < buying_threshold) or (t0_low_increase < buying_threshold) :
                    if (t0_open_increase < buying_threshold) :
                        buying_price = t0_open
                    else:
                        buying_price = pre_close * ((buying_threshold / 100.0) + 1.0)
                    
                    out_price = td_close
                    # for mloop in range(0, (tushare_data.predict_day_count - 1)):
                    #     temp_caption = 'mpred_%d' % mloop
                    #     monitor_pred = result.iloc[iloop][temp_caption]
                    #     if monitor_pred < 0:
                    #         temp_caption = 'T%d_open' % (mloop + 1)
                    #         out_price = result.iloc[iloop][temp_caption]
                    #         break

                    # for mloop in range(0, (tushare_data.predict_day_count - 1)):
                    #     if mloop >= 1:
                    #         temp_caption = 'T%d_close' % mloop
                    #         temp_close = result.iloc[iloop][temp_caption]
                    #         temp_increase = ((temp_close / buying_price) - 1.0) * 100.0
                    #         if temp_increase < (-10.0):
                    #             out_price = temp_close
                    #             break

                    temp_increase = ((out_price / buying_price) - 1.0) *100.0
                    day_increase_sum = day_increase_sum + temp_increase
                    day_trade_count = day_trade_count + 1
                    
                    increase_sum = increase_sum + temp_increase
                    trade_count = trade_count + 1
                
                    print("%10u%10u%10u    %06u%10.2f%10.2f%10.2f%10.2f%10.2f%10.2f%10.2f%10.2f%10.2f" %( \
                        trade_count, \
                        int(in_trade_date), \
                        int(out_trade_date), \
                        int(stock_code), \
                        pred, \
                        t0_open_increase, \
                        t0_low_increase, \
                        t0_open, \
                        t0_low, \
                        buying_price, \
                        out_price, \
                        td_close, \
                        temp_increase))
    if day_trade_count > 0 :
        day_avg_increase=day_increase_sum/day_trade_count





if trade_count > 0:
    avg_increase=increase_sum/trade_count
else:
    avg_increase=0.0
print("Global, trade_count=%u, ave_increase=%f" % (trade_count, avg_increase))



# result_sum = result_sum.copy()
# result_sum = result_sum.reset_index(drop=True)
# result_sum['act_increase'] = 0.0
# label_sum = 0.0
# for iloop in range(0, len(result_sum)):
#     pre_close = result_sum.iloc[iloop]['pre_close']
#     t0_open_increase = result_sum.iloc[iloop]['T0_open_increse']
#     t0_low_increase = result_sum.iloc[iloop]['T0_low_increase']
#     t0_open = result_sum.iloc[iloop]['T0_open']
#     t0_low = result_sum.iloc[iloop]['T0_low']
#     td_close = result_sum.iloc[iloop]['Td_close']
#     buying_threshold = 9.0
#     if (t0_open_increase < buying_threshold) or (t0_low_increase < buying_threshold) :
#         if (t0_open_increase < buying_threshold) :
#             buying_price = t0_open
#         else:
#             buying_price = pre_close * ((buying_threshold / 100.0) + 1.0)
#         result_sum.loc[iloop,'act_increase'] = ((td_close / buying_price) - 1.0) *100.0

# result_sum = result_sum.sort_values(by='act_increase', ascending=False)
# result_sum.to_csv('./test_result_sum_sort.csv')
