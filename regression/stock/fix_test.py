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
import train_rnn
import fix_dataset
import feature

reload(sys)
sys.setdefaultencoding('utf-8')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

predict_trade_threshold = 4
max_trade_count_1_day = 1
test_data_start_date = 20190414

def Predict(test_data, model, mean, std):
    predict_features = test_data[:, feature.COL_FEATURE_OFFSET(): feature.COL_FEATURE_OFFSET() + feature.FEATURE_SIZE()]
    predict_features = train_rnn.FeaturesPretreat(predict_features, mean, std)
    predictions = model.predict(predict_features)
    predictions_df = pd.DataFrame(predictions, columns=['pred'])

    temp_index = feature.COL_ACTURE_OFFSET(0)
    acture_unit_size = feature.ACTURE_UNIT_SIZE()
    t0_acture_data = test_data[:, temp_index: temp_index+acture_unit_size]
    acture_data_df = pd.DataFrame(t0_acture_data, columns=[ \
        'T0_open_increse', \
        'T0_low_increase', \
        'T0_open', \
        'T0_low', \
        'T0_close', \
        'stock_code', \
        'T0_trade_date'])
        
    temp_index = feature.COL_ACTURE_OFFSET(feature.ACTIVE_LABEL_DAY())
    td_acture_data = test_data[:, temp_index: temp_index+acture_unit_size]
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
    result_all = pd.merge(result_all, acture_data_df, left_index=True, right_index=True)
    return result_all

def TestEntry(test_data, print_msg, model, mean, std):
    result_all = Predict(test_data, model, mean, std)
    trade_count = 0
    capital_ratio = 1.0
    capital_value = 1.0
    increase_sum = 0.0
    init_flag = True
    if print_msg:
        print("%6s%10s%10s%8s%6s%16s%16s%8s%8s%8s%8s" %(
            "index", \
            "indate", \
            "outdate", \
            "code", \
            "pred", \
            "t0_open", \
            "t0_low", \
            "buy", \
            "out", \
            "act_inc", \
            "capital"))
        print("-------------------------------------------------------------------------------")
    # 根据 t0_trade_date 生成 date_list
    test_date_list = np.unique(result_all['T0_trade_date'].values).tolist()
    for trade_date in test_date_list:
        result = result_all[result_all['T0_trade_date'] == float(trade_date)]
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
                    t0_open_increase = result.iloc[iloop]['T0_open_increse']
                    t0_low_increase = result.iloc[iloop]['T0_low_increase']
                    t0_open = result.iloc[iloop]['T0_open']
                    t0_low = result.iloc[iloop]['T0_low']
                    td_close = result.iloc[iloop]['Td_close']
                    td_open = result.iloc[iloop]['Td_open']
                    pre_close = t0_open / (t0_open_increase / 100.0 + 1.0)
                    if feature.label_type == feature.LABEL_PRE_CLOSE_2_TD_CLOSE:
                        buying_threshold = pred - 5.0
                        if buying_threshold > 9.0 :
                            buying_threshold = 9.0
                    else:
                        buying_threshold = 9.0
                    
                    if pred > predict_trade_threshold :
                        if (t0_open_increase < buying_threshold) or (t0_low_increase < buying_threshold) :
                            if (t0_open_increase < buying_threshold) :
                                buying_price = t0_open
                            else:
                                buying_price = pre_close * ((buying_threshold / 100.0) + 1.0)
                            
                            if feature.label_type == feature.LABEL_T1_OPEN_2_TD_OPEN:
                                out_price = td_open
                            else:
                                out_price = td_close
                            
                            # 监控
                            # if tushare_data.test_acture_data_with_feature:
                            #     for mloop in range(0, (tushare_data.predict_day_count - 1)):
                            #         temp_caption = 'mpred_%d' % mloop
                            #         monitor_pred = result.iloc[iloop][temp_caption]
                            #         if monitor_pred < 0:
                            #             temp_caption = 'T%d_open' % (mloop + 1)
                            #             out_price = result.iloc[iloop][temp_caption]
                            #             break

                            # 止损
                            # for mloop in range(0, (tushare_data.predict_day_count - 1)):
                            #     if mloop >= 1:
                            #         temp_caption = 'T%d_close' % mloop
                            #         temp_close = result.iloc[iloop][temp_caption]
                            #         temp_increase = ((temp_close / buying_price) - 1.0) * 100.0
                            #         if temp_increase < (0.0):
                            #             out_price = temp_close
                            #             break

                            temp_increase = ((out_price / buying_price) - 1.0 - 0.001) *100.0
                            increase_sum += temp_increase

                            day_trade_count += 1
                            trade_count += 1
                            # print("%f, %f, %f" % (capital_value, capital_ratio, temp_increase))
                            capital_value += capital_ratio / float(tushare_data.predict_day_count) / float(max_trade_count_1_day) * (temp_increase / 100.0)

                            if capital_value > 1.0:
                                capital_ratio = int(capital_value)  # 每增加1倍更新一次

                            if print_msg:
                                print("%6u%10u%10u  %06u%6.2f%16s%16s%8.2f%8.2f%8.2f%8.4f" %( \
                                    trade_count, \
                                    int(in_trade_date), \
                                    int(out_trade_date), \
                                    int(stock_code), \
                                    pred, \
                                    "%5.2f|%5.2f" % (t0_open, t0_open_increase), \
                                    "%5.2f|%5.2f" % (t0_low, t0_low_increase), \
                                    buying_price, \
                                    out_price, \
                                    temp_increase, \
                                    capital_value))





    if trade_count > 0:
        avg_increase = increase_sum / trade_count
    else:
        avg_increase = 0.0
    capital_increase = (capital_value - 1.0) * 100.0
    if print_msg:
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

if __name__ == "__main__":
    # max_capital_increase = -10000
    # max_capital_increase_threshold = 0
    # max_capital_increase_max_trade_count_1_day = 0
    print("%16s%16s%16s%16s%16s" %(
                "in_thre", \
                "trade_1_day", \
                "trade_count", \
                "ave_increase", \
                "capital_increase"))
    print("-------------------------------------------------------------------------------")
    # for threshold in range(0, 10):
    #     for temp_count in range(1, 2):
    #         temp_capital_increase = TestEntry(threshold, temp_count, False)
    #         if temp_capital_increase > max_capital_increase:
    #             max_capital_increase = temp_capital_increase
    #             max_capital_increase_threshold = threshold
    #             max_capital_increase_max_trade_count_1_day = temp_count
    # print("max:")
    # TestEntry(max_capital_increase_threshold, max_capital_increase_max_trade_count_1_day, True)
    dataset_name = 'fix'
    model_epoch = -1
    if len(sys.argv) > 1:
        dataset_name = sys.argv[1]
    if len(sys.argv) > 2:
        model_epoch = int(sys.argv[2])

    if dataset_name == 'daily':
        test_data = fix_dataset.GetDailyDataSet(test_data_start_date)
    else:
        test_data = fix_dataset.GetTestData()
    model, mean, std = train_rnn.LoadModel('fix', model_epoch)
    TestEntry(test_data, True, model, mean, std)


