# -*- coding:UTF-8 -*-
import time
import os
import threading
import multiprocessing
import Queue
from const_def import *

def TradeDateStr(pre_trade_date, trade_date):
    if type(trade_date) == str:
        return trade_date
    elif trade_date != INVALID_DATE:
        return ('%d' % int(trade_date))
    elif type(pre_trade_date) == str:
        return ('%s+1' % pre_trade_date)
    elif pre_trade_date != INVALID_DATE:
        return ('%d+1' % int(pre_trade_date))
    else:
        return '--'

def CodeStr(ts_code):
    if type(ts_code) == str:
        return ts_code
    else:
        return '%06u' % int(ts_code)

def PriceStr(price):
    if price <= 0.0:
        return '--'
    else:
        return '%.2f' % price

def IncreaseStr(on_price, off_price, short_trade=False):
    if on_price == 0 or off_price == 0:
        return '--'
    else:
        if short_trade:
            return '%.4f' % (on_price / off_price - 1.0)
        else:
            return '%.4f' % (off_price / on_price - 1.0)

def IntStr(input_data):
    if type(input_data) == str:
        return input_data
    else:
        return '%u' % int(input_data)

def FloatStr(input_data):
    if type(input_data) == str:
        return input_data
    else:
        if abs(input_data) < 1000:
            return '%.2f' % float(input_data)
        else:
            return '%.2E' % float(input_data)

def IncPct(x, x_base):
    return (x - x_base) / x_base * 100.0

# ----------------------------------------------------------------------------------
# Usage:
#       PrintTrade(1, '000001.SZ', ...)
#       PrintTrade(1, 600050.0, ...)
#       PrintTrade(1, 600050.0, INVALID_DATE, 20190102.0, 20190103.0, INVALID_DATE, ...)
#       PrintTrade('Sum', '--', '--', '--', '--', '--', ...)
# ----------------------------------------------------------------------------------
def PrintTrade(trade_index, 
               ts_code, 
               pre_on_date, on_date, 
               pre_off_date, off_date, 
               increase, holding_days, prediction=None):
    
    if prediction == None:
        print("%-6s%-10s%-12s%-12s%-10s%-10s" %( \
                IntStr(trade_index), \
                CodeStr(ts_code), \
                TradeDateStr(pre_on_date, on_date), \
                TradeDateStr(pre_off_date, off_date), \
                FloatStr(increase), \
                IntStr(holding_days)))
    else:
        print("%-6s%-10s%-10s%-12s%-12s%-10s%-10s" %( \
                IntStr(trade_index), \
                CodeStr(ts_code), \
                FloatStr(prediction), \
                TradeDateStr(pre_on_date, on_date), \
                TradeDateStr(pre_off_date, off_date), \
                FloatStr(increase), \
                IntStr(holding_days)))

def CurrentDate():
    return time.strftime('%Y%m%d',time.localtime(time.time()))

def MKFileDirs(file_name):
    temp_index = file_name.rfind('/')
    if temp_index > 0:
        path_name = file_name[:temp_index]
        if not os.path.exists(path_name):
            print('os.makedirs(%s)' % path_name)
            os.makedirs(path_name)

def ListToIndexMap(input_list, to_int_value=False):
    temp_dict = {}
    for iloop in range(0, len(input_list)):
        if to_int_value:
            temp_dict[int(input_list[iloop])] = iloop
        else:
            temp_dict[input_list[iloop]] = iloop
    return temp_dict

def ListToMap(list1, list2):
    temp_dict = {}
    for iloop in range(0, len(list1)):
        temp_dict[list1[iloop]] = list2[iloop]
    return temp_dict

def ListMultiThreadFunc(func, param, msg_q):
    print('ListMultiThreadFunc')
    while(True):
        try:
            msg = msg_q.get_nowait()
        except:
            break
        func(param, msg)
        
def ListMultiThread(func, param, thread_num, data_list):
    msg_q = multiprocessing.Queue(-1)
    for iloop in data_list:
        msg_q.put(iloop)

    t_list = []
    for iloop in range(thread_num):
        t = multiprocessing.Process(target = ListMultiThreadFunc, args=(func, param, msg_q))
        # t = threading.Thread(target = ListMultiThreadFunc, 
        #                      name='ListMultiThread_%u' % iloop,
        #                      args=(func, param, msg_q))
        t.start()
        t_list.append(t)
    for iloop in range(thread_num):
        t_list[iloop].join()
    
def ImportMatPlot(use_agg=False):
    import matplotlib
    font_name = r"/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"
    if use_agg:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdate
    from matplotlib.widgets import Cursor
    from matplotlib.font_manager import FontProperties
    import matplotlib.lines as lines
    zhfont = FontProperties(fname=font_name, size=15)
    return plt, mdate, Cursor, lines, zhfont

# def UserFunc(param, msg):
#     print('{}, {}'.format(param, msg))
#     time.sleep(1)

# temp_list = [1, 2, 3, 4, 5]
# ListMultiThread(UserFunc, 'param', 8, temp_list)   

def RmDir(path):
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):
            RmDir(c_path)
        else:
            os.remove(c_path)
    os.rmdir(path)

def ShowDFDTypes(df):
    print(df.dtypes)