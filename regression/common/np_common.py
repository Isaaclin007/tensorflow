# -*- coding:UTF-8 -*-


import numpy as np
import pandas as pd
import os
import time
import datetime
import sys
import math
sys.path.append("..")
from common import base_common

# 按行排序
def Sort2D(np_data, sort_index_list, ascending_order_list=[]):
    captions = []

    for iloop in range(np_data.shape[1]):
        captions.append(str(iloop))

    order_list = []
    dst_ascending_order_list = []

    for iloop in range(len(sort_index_list)):
        order_list.append(str(sort_index_list[iloop]))
        
    if len(ascending_order_list) == 0:
        for iloop in range(len(sort_index_list)):
            dst_ascending_order_list.append(True)
    else:
        dst_ascending_order_list = ascending_order_list

    data_df = pd.DataFrame(np_data, columns=captions)
    data_df = data_df.sort_values(by=order_list, ascending=dst_ascending_order_list)
    return data_df.values

def RandSelect(np_data, select_num):
    rand_array = np.arange(np_data.shape[0])
    np.random.shuffle(rand_array)
    return np_data[rand_array[0:select_num]]

def ShowHist(np_data, step = 1, drop_ratio = 0.001):
    min_data = np.min(np_data)
    max_data = np.max(np_data)
    bins = np.arange(min_data - step, max_data + 2 * step, step)

    np_hist = np.histogram(np_data, bins = bins)
    drop_data_count = drop_ratio * len(np_data)
    temp_count = 0
    for iloop in range(len(bins) - 1):
        temp_count += np_hist[0][iloop]
        if temp_count > drop_data_count:
            min_data = np_hist[1][iloop]
            break
    temp_count = 0
    for iloop in reversed(range(len(bins) - 1)):
        temp_count += np_hist[0][iloop]
        if temp_count > drop_data_count:
            max_data = np_hist[1][iloop]
            break
    # bins = np.arange(min_data, max_data + step, step)
    bins = np.arange(min_data - step, max_data + 2 * step, step)
    np_hist = np.histogram(np_data, bins = bins)
    print('%-16s%-16s' % ('bins', 'num'))
    print('*' * 32)
    for iloop in range(len(bins) - 1):
        print('%-16f%-16u' % (np_hist[1][iloop], np_hist[0][iloop]))

    plt, mdate, zhfont = base_common.ImportMatPlot()
    plt.hist(np_data, bins = bins)
    plt.title("histogram") 
    plt.show()

def on_key(event, key_value):
    key_value.insert(0, event.key)

# data_list: np 2D data list, (x, y), data unit shape=(len, 2)
def Show2DData(title, data_list, name_list, x_is_date = False):
    plt, mdate, zhfont = base_common.ImportMatPlot()
    plt.ion()
    title = unicode(title, "utf-8")
    fig1 = plt.figure(dpi=70,figsize=(32,10))

    key_value = []
    fig1.canvas.mpl_connect('key_press_event', lambda event: on_key(event, key_value))

    ax1 = fig1.add_subplot(1,1,1) 
    ax1.xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m-%d'))
    plt.title(title, fontproperties=zhfont)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    for iloop in range(len(data_list)):
        x = data_list[iloop][:, 0]
        y = data_list[iloop][:, 1]
        if len(name_list) > iloop:
            name = name_list[iloop]
        else:
            name = str(iloop)
        if x_is_date:
            xs = [datetime.datetime.strptime(d, '%Y%m%d').date() for d in x.astype(int).astype(str)]
        else:
            xs = x
        plt.plot(xs, y, label=name, linewidth=1)
    plt.gcf().autofmt_xdate()
    plt.legend()
    plt.show()
    while(1):
        plt.pause(0.5)
        if len(key_value) > 0:
            break
    # plt.pause(1)
    plt.close()
    if key_value[0] == 'escape':
        exit()


def Grad(np_data):
    data_len = len(np_data)
    grad_data = np.zeros(data_len)
    grad_data[:data_len-1] = np_data[:data_len-1] - np_data[1:]
    return grad_data