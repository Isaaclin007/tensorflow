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
import matplotlib.pyplot as plt
import matplotlib as mpl
from datetime import datetime 
import matplotlib.dates as mdate
from matplotlib.font_manager import FontProperties
zhfont = FontProperties(fname=r"/usr/share/fonts/truetype/wqy/wqy-microhei.ttc", size=15)
import pp_daily_update
import avg_wave
from common.base_common import *

reload(sys)
sys.setdefaultencoding('utf-8')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

# mpl.rcParams['font.sans-serif']=['SimHei'] #指定默认字体 SimHei为黑体
# mpl.rcParams['axes.unicode_minus']=False #用来正常显示负号

def GetProprocessedData(ts_code):
    stock_pp_file_name = tushare_data.FileNameStockPreprocessedData(ts_code)
    if os.path.exists(stock_pp_file_name):
        pp_data = pd.read_csv(stock_pp_file_name)
        return pp_data
    else:
        print("File not exist: %s" % stock_pp_file_name)
        return []

def PlotCondition(pp_data, data_name, data_value, input_color):
    if len(pp_data) == 0:
        return
    filter_data = pp_data[pp_data[data_name] == data_value]
    if len(filter_data) == 0:
        return
    xs = [datetime.strptime(d, '%Y%m%d').date() for d in filter_data['trade_date'].astype(str).values]
    plt.plot(xs, filter_data['close'].values, 'o', color=input_color, label='%s_%u' % (data_name, data_value), linewidth=1)

def ShowAStock(ts_code, show_values=['open', 
                                     'close', 
                                     'close_5_avg', 
                                     'close_10_avg', 
                                     'close_30_avg',
                                     'close_100_avg',
                                     'vol',
                                     'vol_5_avg',
                                     'vol_10_avg',
                                     'vol_30_avg',
                                     'wave']):
    pp_data = GetProprocessedData(ts_code)
    # pp_data = pp_data[:500]
    if len(pp_data) == 0:
        return
    # pp_data = pp_daily_update.GetPreprocessedDataExt(ts_code)
    stock_name = tushare_data.StockName(ts_code)
    title = "%s - %s" % (ts_code, stock_name)
    title = unicode(title, "utf-8")
    fig1 = plt.figure(dpi=70,figsize=(32,10))
    ax1 = fig1.add_subplot(1,1,1) 
    ax1.xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m-%d'))
    plt.title(title, fontproperties=zhfont)
    plt.xlabel('date')
    plt.ylabel('price')
    xs = [datetime.strptime(d, '%Y%m%d').date() for d in pp_data['trade_date'].astype(str).values]
    plt.grid(True)

    # plt.plot(xs, pp_data['open'].values, label='open', linewidth=1)
    # plt.plot(xs, pp_data['close'].values, label='close', linewidth=2)
    # plt.plot(xs, pp_data['close_5_avg'].values, label='5', linewidth=1)
    # plt.plot(xs, pp_data['close_10_avg'].values, label='10', linewidth=1)
    # plt.plot(xs, pp_data['close_30_avg'].values, label='30', linewidth=1)
    # plt.plot(xs, pp_data['close_100_avg'].values, label='100', linewidth=1)
    # plt.plot(xs, pp_data['close_100_avg'].values, label='100', linewidth=1)

    # pp_data['vol'] = pp_data['vol'] / pp_data.loc[0,'vol_100_avg'] * pp_data.loc[0,'close_100_avg'] * 0.2
    close_max = pp_data['close'].max()
    vol_ratio = 1.0 / pp_data['vol'].max() * close_max / 2
    pp_data['vol'] = pp_data['vol'] * vol_ratio
    pp_data['vol_5_avg'] = pp_data['vol_5_avg'] * vol_ratio
    pp_data['vol_10_avg'] = pp_data['vol_10_avg'] * vol_ratio
    pp_data['vol_30_avg'] = pp_data['vol_30_avg'] * vol_ratio
    for name in show_values:
        if name == 'wave':
            wave_kernel.AppendWaveData(pp_data)
            PlotCondition(pp_data, 'wave_extreme', wave_kernel.EXTREME_PEAK)
            PlotCondition(pp_data, 'wave_extreme', wave_kernel.EXTREME_VALLEY)
        elif name == 'avg_wave':
            avg_wave.AppendWaveData(pp_data)
            print(pp_data)
            PlotCondition(pp_data, 'avg_wave_status', WS_UP, 'r')
            PlotCondition(pp_data, 'avg_wave_status', WS_DOWN, 'black')
        else:
            plt.plot(xs, pp_data[name].values, label=name, linewidth=1)
    # plt.plot(xs, pp_data['vol'].values, label='vol', linewidth=1)
    # plt.plot(xs, pp_data['vol_5_avg'].values, label='v_5', linewidth=1)

    plt.gcf().autofmt_xdate()
    plt.legend()
    #plt.ylim([0,5])
    plt.show()

if __name__ == "__main__":
    # ShowAStock('600050.SH')
    if len(sys.argv) > 1:
        # ShowAStock(sys.argv[1])
        ShowAStock(sys.argv[1], ['close', 'close_30_avg', 'wave'])
    else:
        code_list = ['000001.SZ']
        for ts_code in code_list:
            ShowAStock(ts_code, ['close', 'close_30_avg', 'avg_wave'])

    



