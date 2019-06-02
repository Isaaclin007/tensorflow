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


def ShowAStock(ts_code):
    tushare_data.train_test_date = tushare_data.CurrentDate()
    tushare_data.DownloadAStocksData(ts_code)
    tushare_data.UpdatePreprocessDataAStock(0, ts_code)
    pp_data = GetProprocessedData(ts_code)
    stock_name = tushare_data.StockName(ts_code)
    title = "%s - %s" % (ts_code, stock_name)
    title = unicode(title, "utf-8")
    fig1 = plt.figure(dpi=70,figsize=(32,10))
    ax1 = fig1.add_subplot(1,1,1) 
    ax1.xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m-%d'))
    plt.title(title, fontproperties=zhfont)
    # plt.title(u"中文", fontproperties=zhfont)
    # plt.subplot(1, 1, 1)
    plt.xlabel('date')
    plt.ylabel('price')
    xs = [datetime.strptime(d, '%Y%m%d').date() for d in pp_data['trade_date'].astype(str).values]
    # plt.xticks(xs,rotation=45)
    # xs = pp_data['trade_date']
    plt.grid(True)

    # plt.plot(xs, pp_data['open'].values, label='open', linewidth=1)
    plt.plot(xs, pp_data['close'].values, label='close', linewidth=2)
    # plt.plot(xs, pp_data['close_5_avg'].values, label='5', linewidth=1)
    plt.plot(xs, pp_data['close_10_avg'].values, label='10', linewidth=1)
    # plt.plot(xs, pp_data['close_30_avg'].values, label='30', linewidth=1)
    # plt.plot(xs, pp_data['close_100_avg'].values, label='100', linewidth=1)
    plt.plot(xs, pp_data['close_100_avg'].values, label='100', linewidth=1)

    # pp_data['vol'] = pp_data['vol'] / pp_data.loc[0,'vol_100_avg'] * pp_data.loc[0,'close_100_avg'] * 0.2
    close_max = pp_data['close'].max()
    vol_ratio = 1.0 / pp_data['vol'].max() * close_max / 2
    pp_data['vol'] = pp_data['vol'] * vol_ratio
    pp_data['vol_5_avg'] = pp_data['vol_5_avg'] * vol_ratio
    pp_data['vol_10_avg'] = pp_data['vol_10_avg'] * vol_ratio
    pp_data['vol_30_avg'] = pp_data['vol_30_avg'] * vol_ratio
    pp_data['vol_200_avg'] = pp_data['vol_200_avg'] * vol_ratio
    # pp_data['sell_sm_vol_30_avg'] = (pp_data['sell_sm_vol_30_avg'] - pp_data['buy_sm_vol_30_avg']) * vol_ratio * 10.0
    # pp_data['sell_elg_vol_30_avg'] = (pp_data['sell_elg_vol_30_avg'] - pp_data['buy_elg_vol_30_avg']) * vol_ratio * 10.0
    plt.plot(xs, pp_data['vol'].values, label='vol', linewidth=1)
    plt.plot(xs, pp_data['vol_5_avg'].values, label='v_5', linewidth=1)
    # plt.plot(xs, pp_data['vol_10_avg'].values, label='v_10', linewidth=1)
    # plt.plot(xs, pp_data['vol_200_avg'].values, label='v_200', linewidth=1)
    # plt.plot(xs, pp_data['sell_elg_vol_30_avg'].values, label='ssm_vol', linewidth=1)

    # wave_kernel.AppendWaveData(pp_data)
    # peak_data = pp_data[pp_data['wave_extreme'] == wave_kernel.EXTREME_PEAK]
    # xs = [datetime.strptime(d, '%Y%m%d').date() for d in peak_data['trade_date'].astype(str).values]
    # plt.plot(xs, peak_data[wave_kernel.wave_index].values, label='peak', linewidth=1)

    # valley_data = pp_data[pp_data['wave_extreme'] == wave_kernel.EXTREME_VALLEY]
    # xs = [datetime.strptime(d, '%Y%m%d').date() for d in valley_data['trade_date'].astype(str).values]
    # plt.plot(xs, valley_data[wave_kernel.wave_index].values, label='valley', linewidth=1)

    plt.gcf().autofmt_xdate()
    plt.legend()
    #plt.ylim([0,5])
    plt.show()

if __name__ == "__main__":
    # ShowAStock('600050.SH')
    if len(sys.argv) > 1:
        ShowAStock(sys.argv[1])
    else:
        code_list = ['002236.SZ', '002415.SZ', '000650.SZ', '000937.SZ', '600104.SH']
        for ts_code in code_list:
            ShowAStock(ts_code)

    



