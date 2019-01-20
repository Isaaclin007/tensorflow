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
feature_size = tushare_data.feature_size
acture_size = tushare_data.acture_size
test_data = tushare_data.GetTestData()
test_date_list = tushare_data.TestTradeDateList()
# test_date_list = test_date_list[0:100]
# print("len(test_date_list):")
# print(len(test_date_list))
# load_data = np.load('./temp_data/test_data_20090101_20190104_100_1_1_0.npy')

temp_index = tushare_data.TestDataLastPredictFeatureOffset()
predict_features = test_data[:, temp_index: temp_index + feature_size]
predict_features = (predict_features - mean) / std
predictions = model.predict(predict_features)
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
current_data = test_data[:, temp_index: temp_index+1]
current_data_df = pd.DataFrame(current_data, columns=['pre_close'])

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

result_all = predictions_df
result_all = pd.merge(result_all, current_data_df, left_index=True, right_index=True)
result_all = pd.merge(result_all, acture_data_df, left_index=True, right_index=True)

def TestEntry(predict_trade_threshold, max_trade_count_1_day, print_msg):
    trade_count = 0
    capital_ratio = 1.0
    capital_value = 1.0
    increase_sum = 0.0
    init_flag = True
    if print_msg:
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
    for trade_date in reversed(test_date_list):
        result = result_all[result_all['Td_trade_date'] == float(trade_date)]
        if len(result) > 0:
            result = result.sort_values(by = 'pred', ascending=False)
            # result.to_csv(('./result_%d_sort.csv' % day_loop))
            result = result[:20]
            if init_flag:
                init_flag = False
                result_sum = result
            else:
                result_sum = result_sum.append(result)

            day_increase_sum=0.0
            day_trade_count=0
            day_avg_increase=0.0

            for iloop in range(0, len(result)):
                if day_trade_count < max_trade_count_1_day :
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
                    if tushare_data.label_type == tushare_data.LABEL_PRE_CLOSE_2_TD_CLOSE:
                        buying_threshold = pred - 5.0
                        if buying_threshold > 9.0 :
                            buying_threshold = 9.0
                    elif tushare_data.label_type == tushare_data.LABEL_T1_OPEN_2_TD_CLOSE:
                        buying_threshold = 9.0
                    if pred > predict_trade_threshold :
                        if (t0_open_increase < buying_threshold) or (t0_low_increase < buying_threshold) :
                            if (t0_open_increase < buying_threshold) :
                                buying_price = t0_open
                            else:
                                buying_price = pre_close * ((buying_threshold / 100.0) + 1.0)
                            
                            out_price = td_close
                            # 监控
                            # for mloop in range(0, (tushare_data.predict_day_count - 1)):
                            #     temp_caption = 'mpred_%d' % mloop
                            #     monitor_pred = result.iloc[iloop][temp_caption]
                            #     if monitor_pred < 0:
                            #         temp_caption = 'T%d_open' % (mloop + 1)
                            #         out_price = result.iloc[iloop][temp_caption]
                            #         break

                            # 止损
                            # for mloop in range(0, (tushare_data.predict_day_count - 1)):
                            #     if mloop >= 1:
                            #         temp_caption = 'T%d_close' % mloop
                            #         temp_close = result.iloc[iloop][temp_caption]
                            #         temp_increase = ((temp_close / buying_price) - 1.0) * 100.0
                            #         if temp_increase < (-5.0):
                            #             out_price = temp_close
                            #             break

                            temp_increase = ((out_price / buying_price) - 1.0) *100.0
                            increase_sum += temp_increase
                        
                            if print_msg:
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

                            day_trade_count += 1
                            trade_count += 1
                            # print("%f, %f, %f" % (capital_value, capital_ratio, temp_increase))
                            capital_value += capital_ratio / float(tushare_data.predict_day_count) / float(max_trade_count_1_day) * ((out_price / buying_price) - 1.0)

                            if capital_value > 1.0:
                                capital_ratio = int(capital_value)  # 每增加1倍更新一次





    if trade_count > 0:
        avg_increase = increase_sum / trade_count
    else:
        avg_increase = 0.0
    capital_increase = (capital_value - 1.0) * 100.0
    print("%16u%16u%16u%16.2f%16.2f" % \
        (predict_trade_threshold, max_trade_count_1_day, trade_count, avg_increase, capital_increase))
    return capital_increase


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

max_capital_increase = -10000
max_capital_increase_threshold = 0
max_capital_increase_max_trade_count_1_day = 0
print("%16s%16s%16s%16s%16s" %(
            "in_thre", \
            "trade_1_day", \
            "trade_count", \
            "ave_increase", \
            "capital_increase"))
print("-------------------------------------------------------------------------------")
# for threshold in range(0, 15):
#     for temp_count in range(1, 5):
#         temp_capital_increase = TestEntry(threshold, temp_count, False)
#         if temp_capital_increase > max_capital_increase:
#             max_capital_increase = temp_capital_increase
#             max_capital_increase_threshold = threshold
#             max_capital_increase_max_trade_count_1_day = temp_count
# print("max:")
# TestEntry(max_capital_increase_threshold, max_capital_increase_max_trade_count_1_day, True)
TestEntry(0, 1, True)


