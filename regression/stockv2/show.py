# -*- coding:UTF-8 -*-

import tensorflow as tf
from tensorflow import keras
import tushare as ts
import numpy as np
import pandas as pd
import os
import time
import sys
import random
from datetime import datetime 
sys.path.append("..")
from common import base_common
from common import np_common
from common.const_def import *
import tushare_data
import preprocess

# reload(sys)
# sys.setdefaultencoding('utf-8')
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

# mpl.rcParams['font.sans-serif']=['SimHei'] #指定默认字体 SimHei为黑体
# mpl.rcParams['axes.unicode_minus']=False #用来正常显示负号


def PlotCondition(pp_data, data_name, data_value, input_color):
    if len(pp_data) == 0:
        return
    filter_data = pp_data[pp_data[data_name] == data_value]
    if len(filter_data) == 0:
        return
    xs = [datetime.strptime(d, '%Y%m%d').date() for d in filter_data['trade_date'].astype(str).values]
    plt.plot(xs, filter_data['close'].values, 'o', color=input_color, label='%s_%u' % (data_name, data_value), linewidth=1)

def ShowAStock(ts_code, show_values=[PPI_open, 
                                     PPI_close, 
                                     PPI_close_5_avg, 
                                     PPI_close_10_avg, 
                                     PPI_close_30_avg,
                                     PPI_close_100_avg,
                                     PPI_vol,
                                     PPI_vol_10_avg,
                                     PPI_vol_100_avg]):
    plt, mdate, zhfont = base_common.ImportMatPlot()
    # current_date_int = int(base_common.CurrentDate())
    current_date_int = 20200403
    o_data_source = tushare_data.DataSource(current_date_int, 
                                            '', 
                                            '', 
                                            1, 
                                            0, 
                                            current_date_int)
    o_data_source.DownloadStockData(ts_code)
    o_data_source.UpdateStockPPData(ts_code)
    pp_data = o_data_source.LoadStockPPData(ts_code)
    preprocess.PPRegionCompute(pp_data, PPI_close, PPI_close_5_avg, 500, len(pp_data), np.mean)
    stock_name = o_data_source.StockName(ts_code)
    if len(pp_data) == 0:
        return
    title = "%s - %s" % (ts_code, stock_name)
    title = unicode(title, "utf-8")
    fig1 = plt.figure(dpi=70,figsize=(32,10))
    ax1 = fig1.add_subplot(1,1,1) 
    ax1.xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m-%d'))
    plt.title(title, fontproperties=zhfont)
    plt.xlabel('date')
    plt.ylabel('price')
    date_str_arr = np.array(["%.0f" % x for x in pp_data[:,PPI_trade_date]])
    xs = [datetime.strptime(d, '%Y%m%d').date() for d in date_str_arr]
    plt.grid(True)

    vol_list = [PPI_vol,
                PPI_vol_5_avg,
                PPI_vol_10_avg,
                PPI_vol_30_avg,
                PPI_vol_100_avg]
    print(set(show_values) - set(vol_list))
    if len(set(show_values) - set(vol_list)) > 0:
        close_max = max(pp_data[:,PPI_close])
        vol_max = max(pp_data[:,PPI_vol])
        vol_ratio = 1.0 / vol_max * close_max / 2
    else:
        vol_ratio = 1.0
    for col_index in show_values:
        name = ''
        if col_index in vol_list:
            # plt.bar(xs, pp_data[:, col_index] * vol_ratio, 0.8)
            plt.plot(xs, pp_data[:, col_index] * vol_ratio, label=PPI_name[col_index], linewidth=1)
        else:
            plt.plot(xs, pp_data[:, col_index], label=PPI_name[col_index], linewidth=1)

    plt.gcf().autofmt_xdate()
    plt.legend()
    #plt.ylim([0,5])
    plt.show()

if __name__ == "__main__":
    code_list = ['000001.SZ']
    if len(sys.argv) > 1:
        code_list = [sys.argv[1]]
    for ts_code in code_list:
        # ShowAStock(ts_code, [PPI_close, PPI_close_30_avg, PPI_vol_5_avg, PPI_vol_30_avg])
        ShowAStock(ts_code, [PPI_close, PPI_close_5_avg, PPI_close_10_avg, PPI_close_30_avg, PPI_close_100_avg, PPI_vol_5_avg])

    



