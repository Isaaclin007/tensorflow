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
import wave_kernel
import random
import daily_data
import matplotlib.pyplot as plt
import matplotlib as mpl
from datetime import datetime 
import matplotlib.dates as mdate
from matplotlib.font_manager import FontProperties
zhfont = FontProperties(fname=r"/usr/share/fonts/truetype/wqy/wqy-microhei.ttc", size=15)

reload(sys)
sys.setdefaultencoding('utf-8')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

predict_threshold = 0

model=keras.models.load_model("./model/model.h5")
mean=np.load('./model/mean.npy')
std=np.load('./model/std.npy')

def AvgValue(sum_value, sample_num):
    if sample_num == 0:
        return 0.0
    else:
        return sum_value / sample_num

def RegressionTest(test_data):
    predict_features = test_data[:, 0: tushare_data.feature_size]
    print("predict_features: {}".format(predict_features.shape))
    col_index = tushare_data.feature_size
    labels = test_data[:, col_index: col_index + 1]

    col_index += 1
    ts_codes = test_data[:, col_index: col_index + 1]

    col_index += 1
    on_pretrade_dates = test_data[:, col_index: col_index + 1]

    col_index += 1
    on_dates = test_data[:, col_index: col_index + 1]

    col_index += 1
    off_dates = test_data[:, col_index: col_index + 1]

    col_index += 1
    holding_days = test_data[:, col_index: col_index + 1]

    predict_features = (predict_features - mean) / std
    predictions = model.predict(predict_features)
    trade_count = 0
    increase_sum = 0.0
    holding_days_sum = 0
    max_drawdown = 0.0
    max_increase_sum = 0.0
    for iloop in range(0, len(test_data)):
        if predictions[iloop] > predict_threshold:
        # if True:
        # if labels[iloop] > 10:
        # vol_sum = 0.0
        # for dloop in range(0, 10):
        #     vol_sum += test_data[iloop][5 + dloop * 10]
        # vol_avg = vol_sum / 10
        # vol_sum = 0.0
        # for dloop in range(0, 5):
        #     vol_sum += test_data[iloop][5 + dloop * 10]
        # vol_avg_s = vol_sum / 5
        # if vol_avg_s / vol_avg > 1.2:
        # if True:
            if off_dates[iloop] < 20990101.0:
                increase_sum += labels[iloop]
                holding_days_sum += holding_days[iloop]
                trade_count += 1
                if max_increase_sum < increase_sum:
                    max_increase_sum = increase_sum.copy()
                temp_drawdown = max_increase_sum - increase_sum
                if max_drawdown < temp_drawdown:
                    max_drawdown = temp_drawdown.copy()
            print("%-6u%06u    %-10.0f%-10.0f%-10.0f%-10.0f%-10.2f%-10.2f%-10.2f%-10.2f%-10.2f" %( \
                trade_count, \
                int(ts_codes[iloop]), \
                on_pretrade_dates[iloop], \
                on_dates[iloop], \
                off_dates[iloop], \
                holding_days[iloop], \
                predictions[iloop], \
                labels[iloop], \
                increase_sum, \
                AvgValue(increase_sum, trade_count), \
                AvgValue(increase_sum, holding_days_sum)))
    print("trade_count:%u, increase_sum:%-10.2f, max_drawdown:%.2f" %( \
        trade_count, \
        increase_sum, \
        max_drawdown))
    return

    # 显示持有数量和平均收益率的关系
    date_list = tushare_data.TradeDateListRange('%u' % wave_kernel.train_data_end_date, daily_data.end_date)
    original_avg_increase_list = []
    predicted_avg_increase_list = []
    holding_nums_list = []
    show_date_list = date_list[2:]
    for iloop in range(2, len(date_list)):
        temp_date = date_list[iloop]
        temp_end_date = date_list[iloop - 2]
        pos = (on_dates <= int(temp_date)) & (off_dates > int(temp_date))
        holding_nums = np.sum(pos)
        pos = (on_pretrade_dates >= int(temp_date)) & (on_pretrade_dates <= int(temp_end_date))
        temp_num = np.sum(pos)
        # print('data range: %u, %u' % (int(temp_date), int(temp_end_date)))
        # print('temp_num:%u' % temp_num)
        temp_labels = labels[pos]
        temp_predictions = predictions[pos]
        original_increase_sum = 0.0
        predicted_increase_sum = 0.0
        predicted_passed_num = 0
        for kloop in range(0, temp_num):
            original_increase_sum += temp_labels[kloop]
            if temp_predictions[kloop] > predict_threshold:
                predicted_increase_sum += temp_labels[kloop]
                predicted_passed_num += 1
        original_increase_avg = 0.0
        predicted_increase_avg = 0.0
        if temp_num > 0:
            original_increase_avg = original_increase_sum / temp_num
        if predicted_passed_num > 0:
            predicted_increase_avg = predicted_increase_sum / predicted_passed_num
        holding_nums_list.append(holding_nums / 100.0)
        original_avg_increase_list.append(original_increase_avg)
        predicted_avg_increase_list.append(predicted_increase_avg)
    

    # current_date = 0
    # increase_sum = 0.0
    # increase_sum_stock_num = 0
    # show_date_list = []
    
    # for iloop in range(0, len(test_data)):
    #     if current_date != on_pretrade_dates[iloop]:
    #         if increase_sum_stock_num > 0:
    #             # 输出到list
    #             show_date_list.append('%.0f' % current_date)
    #             avg_increase_list.append(increase_sum / increase_sum_stock_num)
    #             holding_nums_list.append(test_data[iloop][0] / 100)
    #         else:
    #             if current_date > 0:
    #                 show_date_list.append('%.0f' % current_date)
    #                 avg_increase_list.append(0.0)
    #                 holding_nums_list.append(0.0)
    #         current_date = on_pretrade_dates[iloop]
    #         increase_sum = 0.0
    #         increase_sum_stock_num = 0
    #     if predictions[iloop] > 5:
    #         increase_sum += labels[iloop]
    #         increase_sum_stock_num += 1

    title = "holding_nums - avg_increase"
    title = unicode(title, "utf-8")
    fig1 = plt.figure(dpi=70,figsize=(32,10))
    ax1 = fig1.add_subplot(1,1,1) 
    ax1.xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m-%d'))
    plt.title(title, fontproperties=zhfont)
    plt.xlabel('date')
    plt.ylabel('increase')
    xs = [datetime.strptime(d, '%Y%m%d').date() for d in show_date_list]
    plt.grid(True)

    plt.plot(xs, holding_nums_list, label='holding_nums', linewidth=1)
    plt.plot(xs, original_avg_increase_list, label='original_increase', linewidth=1)
    plt.plot(xs, predicted_avg_increase_list, label='predicted_increase', linewidth=1)
    plt.gcf().autofmt_xdate()
    plt.legend()
    plt.show()


def TestMaxTradeOneDay(data_set, predictions, max_trade_one_day):
    trade_count = 0
    increase_sum = 0.0
    holding_days_sum = 0
    max_drawdown = 0.0
    max_increase_sum = 0.0
    
    # 获取 data_set 的最大和最小 pre_on 时间，生成date_list列表
    on_pretrade_dates = data_set[:, wave_kernel.COL_ON_PRETRADE_DATE()]
    start_date = '%.0f' % np.min(on_pretrade_dates)
    end_date = '%.0f' % np.max(on_pretrade_dates)
    date_list = tushare_data.TradeDateListRange(start_date, end_date)
    
    for iloop in reversed(range(0, len(date_list))):
        temp_date = int(date_list[iloop])
        pos = (on_pretrade_dates == temp_date)
        if np.sum(pos) > 0:
            temp_data_set = data_set[pos].copy()
            temp_predictions = predictions[pos].copy()
            sort_index = np.argsort(-temp_predictions[:,0])
            temp_data_set = temp_data_set[sort_index]
            temp_predictions = temp_predictions[sort_index]
            temp_data_set = temp_data_set[: max_trade_one_day]
            temp_predictions = temp_predictions[: max_trade_one_day]
            # print(temp_data_set)

            temp_ts_codes = temp_data_set[:, wave_kernel.COL_TS_CODE()]
            temp_on_pretrade_dates = temp_data_set[:, wave_kernel.COL_ON_PRETRADE_DATE()]
            temp_on_dates = temp_data_set[:, wave_kernel.COL_ON_DATE()]
            temp_off_dates = temp_data_set[:, wave_kernel.COL_OFF_DATE()]
            temp_holding_days = temp_data_set[:, wave_kernel.COL_HOLDING_DAYS()]
            temp_labels = temp_data_set[:, wave_kernel.COL_INCREASE()]
            
            for iloop in range(0, len(temp_data_set)):
                if temp_predictions[iloop] > predict_threshold:
                    if temp_off_dates[iloop] < 20990101.0:
                        increase_sum += temp_labels[iloop]
                        holding_days_sum += temp_holding_days[iloop]
                        trade_count += 1
                        if max_increase_sum < increase_sum:
                            max_increase_sum = increase_sum.copy()
                        temp_drawdown = max_increase_sum - increase_sum
                        if max_drawdown < temp_drawdown:
                            max_drawdown = temp_drawdown.copy()
                    print("%-6u%06u    %-10.0f%-10.0f%-10.0f%-10.0f%-10.2f%-10.2f%-10.2f%-10.2f%-10.2f" %( \
                        trade_count, \
                        int(temp_ts_codes[iloop]), \
                        temp_on_pretrade_dates[iloop], \
                        temp_on_dates[iloop], \
                        temp_off_dates[iloop], \
                        temp_holding_days[iloop], \
                        temp_predictions[iloop], \
                        temp_labels[iloop], \
                        increase_sum, \
                        AvgValue(increase_sum, trade_count), \
                        AvgValue(increase_sum, holding_days_sum)))

def RegressionTestMaxTradeOneDay(test_data, max_trade_one_day):
    predict_features = test_data[:, 0: tushare_data.feature_size]
    print("predict_features: {}".format(predict_features.shape))
    col_index = tushare_data.feature_size
    labels = test_data[:, col_index: col_index + 1]

    col_index += 1
    ts_codes = test_data[:, col_index: col_index + 1]

    col_index += 1
    on_pretrade_dates = test_data[:, col_index: col_index + 1]

    col_index += 1
    on_dates = test_data[:, col_index: col_index + 1]

    col_index += 1
    off_dates = test_data[:, col_index: col_index + 1]

    col_index += 1
    holding_days = test_data[:, col_index: col_index + 1]

    predict_features = (predict_features - mean) / std
    predictions = model.predict(predict_features)
    TestMaxTradeOneDay(test_data, predictions, max_trade_one_day)


if __name__ == "__main__":
    test_data = wave_kernel.GetTestData()
    RegressionTest(test_data)
    # RegressionTestMaxTradeOneDay(test_data, 5)




