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
from absl import app
from absl import flags
from datetime import datetime 
sys.path.append("..")
from common import base_common
from common import np_common
from common.const_def import *
import tushare_data
import preprocess

FLAGS = flags.FLAGS

default_show_values =  [PPI_open, 
                        PPI_close, 
                        PPI_close_5_avg, 
                        PPI_close_10_avg, 
                        PPI_close_30_avg,
                        PPI_close_100_avg,
                        PPI_vol,
                        PPI_vol_10_avg,
                        PPI_vol_100_avg]

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

def ShowAStock_(ts_code, show_values=[PPI_open, 
                                     PPI_close, 
                                     PPI_close_5_avg, 
                                     PPI_close_10_avg, 
                                     PPI_close_30_avg,
                                     PPI_close_100_avg,
                                     PPI_vol,
                                     PPI_vol_10_avg,
                                     PPI_vol_100_avg]):
    plt, mdate, _, _, zhfont = base_common.ImportMatPlot()
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
    plt.pause(1000)

def ShowAStockLowLevel(pp_data, show_obj, show_values, vol_bar=False):
    data_len = len(pp_data)
    if data_len == 0:
        return
    if len(set(show_values) - set(PPI_vol_list)) > 0:
        close_max = max(pp_data[:,PPI_close])
        close_min = min(pp_data[:,PPI_close])
        vol_max = max(pp_data[:,PPI_vol])
        vol_min = 0.0
        vol_range = vol_max - vol_min
        vol_show_range = (close_max - close_min) / 3
        vol_show_offset = close_min - vol_show_range
        show_obj.SetBaseLine(vol_show_offset)
        
        if vol_range < 1.0:
            vol_range = 1.0
        vol_ratio = vol_show_range / vol_range
    else:
        vol_ratio = 1.0
    data_list = []
    name_list = []
    bar_list = []
    for col_index in show_values:
        if col_index in PPI_vol_list:
            data_list.append((pp_data[:, col_index] - vol_min) * vol_ratio + vol_show_offset)
        else:
            data_list.append(pp_data[:, col_index])
        bar_list.append(vol_bar and (col_index == PPI_vol))
        name_list.append(PPI_name[col_index])
    return show_obj.RefreshAlign(pp_data[:,PPI_trade_date], data_list, bar_list, name_list)

# def ShowAStock(pp_data, ts_code, code_name):
#     title = "%s - %s" % (ts_code, code_name)
#     show = np_common.Show(title, True)
#     ShowAStockLowLevel(pp_data, show, show_values=[PPI_open])


# if __name__ == "__main__":
#     code_list = ['000001.SZ']
#     if len(sys.argv) > 1:
#         code_list = [sys.argv[1]]
#     for ts_code in code_list:
#         # ShowAStock(ts_code, [PPI_close, PPI_close_30_avg, PPI_vol_5_avg, PPI_vol_30_avg])
#         ShowAStock(ts_code, [PPI_close, PPI_close_5_avg, PPI_close_10_avg, PPI_close_30_avg, PPI_close_100_avg, PPI_vol_5_avg])

def RandShow(pp_data, show_obj, start_day_index):
    temp_index = start_day_index
    data_len = len(pp_data)
    if data_len == 0:
        return
    show_len = FLAGS.l
    if show_len > data_len:
        show_len = data_len
    while True:
        if (temp_index + show_len) > data_len:
            temp_index = data_len - show_len
        if temp_index < 0:
            temp_index = 0
        ret = ShowAStockLowLevel(pp_data[temp_index:temp_index+show_len], show_obj, [PPI_close, PPI_close_5_avg, PPI_vol], False)
        if 'left' == ret:
            temp_index += 1
        elif 'right' == ret:
            temp_index -= 1
        elif 'n' == ret:
            break
        


def main(argv):
    del argv
    random.seed(123)
    show = np_common.Show(True)
    if FLAGS.r:
        data_source = tushare_data.DataSource(20000101, '', '', 1, 20000101, 20200306)
        while True:
            code_index = random.randint(0, len(data_source.code_list) - 1)
            ts_code = data_source.code_list[code_index]
            print(ts_code)
            code_name = data_source.name_list[code_index]
            pp_data = data_source.LoadStockPPData(ts_code, True)
            if len(pp_data) == 0:
                return
            day_index = random.randint(0, len(pp_data) - 1)
            show.SetTitle('%s - %s' % (ts_code, code_name))
            RandShow(pp_data, show, day_index)

    else:
        data_source = tushare_data.DataSource(0, '', '', 1, 20000101, 20200603)
        ts_code = FLAGS.c
        code_name = data_source.StockName(ts_code)
        data_source.DownloadStockData(ts_code)
        data_source.UpdateStockPPData(ts_code)
        pp_data = data_source.LoadStockPPData(ts_code, True)
        show.SetTitle('%s - %s' % (ts_code, code_name))
        ShowAStockLowLevel(pp_data, show, [PPI_open, PPI_vol])
    
    # ShowAStock_(FLAGS.c)

    
if __name__ == "__main__":
    flags.DEFINE_boolean('r', False, 'rand show')
    flags.DEFINE_string('c', '000001.SZ', 'ts code')
    flags.DEFINE_integer('l', 100, 'rand test show data len')
    app.run(main)


