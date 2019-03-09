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
    tushare_data.DownloadAStocksData(ts_code)
    tushare_data.UpdatePreprocessDataAStock(ts_code)
    pp_data = GetProprocessedData(ts_code)
    stock_name = tushare_data.StockName(ts_code)
    title = "%s - %s" % (ts_code, stock_name)
    title = unicode(title, "utf-8")
    plt.figure(dpi=70,figsize=(32,10))
    plt.title(title, fontproperties=zhfont)
    # plt.title(u"中文", fontproperties=zhfont)
    # plt.subplot(1, 1, 1)
    plt.xlabel('date')
    plt.ylabel('price')
    xs = [datetime.strptime(d, '%Y%m%d').date() for d in pp_data['trade_date'].astype(str).values]
    plt.grid(True)

    # plt.plot(xs, pp_data['open'].values, label='open', linewidth=1)
    plt.plot(xs, pp_data['close'].values, label='close', linewidth=1)
    plt.plot(xs, pp_data['close_5_avg'].values, label='5', linewidth=1)
    plt.plot(xs, pp_data['close_10_avg'].values, label='10', linewidth=1)
    # plt.plot(xs, pp_data['close_30_avg'].values, label='30', linewidth=1)
    # plt.plot(xs, pp_data['close_100_avg'].values, label='100', linewidth=1)
    plt.plot(xs, pp_data['close_200_avg'].values, label='200', linewidth=1)
    plt.gcf().autofmt_xdate()
    plt.legend()
    #plt.ylim([0,5])
    plt.show()

if __name__ == "__main__":
    # ShowAStock('600050.SH')
    if len(sys.argv) > 1:
        ShowAStock(sys.argv[1])
    else:
        for ts_code in tushare_data.StockCodes():
            ShowAStock(ts_code)

    



