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
def Show2DData(title, data_list, name_list=[], x_is_date = False):
    plt, mdate, Cursor, zhfont = base_common.ImportMatPlot()
    plt.ion()
    title = unicode(title, "utf-8")
    fig1 = plt.figure(dpi=70,figsize=(32,10))

    key_value = []
    fig1.canvas.mpl_connect('key_press_event', lambda event: on_key(event, key_value))

    ax1 = fig1.add_subplot(1,1,1) 
    if x_is_date:
        ax1.xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m-%d'))
    plt.title(title, fontproperties=zhfont)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    cursor = Cursor(ax1, useblit=True, color='black', linewidth=1)
    for i in range(1):
        plt.cla()
        for iloop in range(len(data_list)):
            x = data_list[iloop][i:i+100, 0]
            y = data_list[iloop][i:i+100, 1]
            if len(name_list) > iloop:
                name = name_list[iloop]
            else:
                name = str(iloop)
            if x_is_date:
                xs = [datetime.datetime.strptime(d, '%Y%m%d').date() for d in x.astype(int).astype(str)]
            else:
                xs = x
            plt.plot(xs, y, label=name, linewidth=1)
        if x_is_date:
            plt.gcf().autofmt_xdate()
        plt.legend()
        plt.show()
        plt.pause(0.1)
        print(i)
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

# class SnaptoCursor(object):
#     """
#     Like Cursor but the crosshair snaps to the nearest x,y point
#     For simplicity, I'm assuming x is sorted
#     """

#     def __init__(self, ax, x, y):
#         self.ax = ax
#         self.lx = ax.axhline(color='k')  # the horiz line
#         self.ly = ax.axvline(color='k')  # the vert line
#         self.x = x
#         self.y = y
#         # text location in axes coords
#         self.txt = ax.text(0.7, 0.9, '', transform=ax.transAxes)

#     def mouse_move(self, event):

#         if not event.inaxes:
#             return

#         x, y = event.xdata, event.ydata

#         indx = min(np.searchsorted(self.x, [x])[0], len(self.x) - 1)
#         x = self.x[indx]
#         y = self.y[indx]
#         # update the line positions
#         self.lx.set_ydata(y)
#         self.ly.set_xdata(x)

#         self.txt.set_text('x=%1.2f, y=%1.2f' % (x, y))
#         print('x=%1.2f, y=%1.2f' % (x, y))
#         plt.draw()

# class Cursor_(object):
#     def __init__(self, ax):
#         self.ax = ax
#         self.lx = ax.axhline(color='k')  # the horiz line
#         self.ly = ax.axvline(color='k')  # the vert line

#         # text location in axes coords
#         self.txt = ax.text(0.7, 0.9, '', transform=ax.transAxes)

#     def mouse_move(self, event):
#         if not event.inaxes:
#             return

#         x, y = event.xdata, event.ydata
#         # update the line positions
#         self.lx.set_ydata(y)
#         self.ly.set_xdata(x)

#         self.txt.set_text('x=%1.2f, y=%1.2f' % (x, y))
#         plt.draw()


class Show():
    def __init__(self, x_is_date = False):
        self.x_is_date = x_is_date
        self.key_value = []
        self.plt, mdate, Cursor, self.lines, self.zhfont = base_common.ImportMatPlot()
        self.plt.ion()
        self.SetTitle('--')
        self.fig1 = self.plt.figure(dpi=70,figsize=(32,10))
        self.ax1 = self.fig1.add_subplot(1,1,1) 
        # self.ax2 = self.ax1.twinx()

        self.fig1.canvas.mpl_connect('key_press_event', lambda event: self.on_key(event))
        self.fig1.canvas.mpl_connect('motion_notify_event', lambda event: self.mouse_move(event))
        
        if self.x_is_date:
            self.ax1.xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m-%d'))
        
        self.plt.xlabel('x')
        self.plt.ylabel('y')
        self.plt.grid(True)
        # self.cursor = Cursor(self.ax1, useblit=True, color='black', linewidth=1)
        self.base_line_value = None
        

    def on_key(self, event):
        self.key_value = event.key

    def mouse_move(self, event):
        if not event.inaxes:
            return
        x, y = event.xdata, event.ydata
        if self.lx == None:
            self.lx = self.plt.axhline(y, c="r", ls="--", lw=1)
            self.txt = self.plt.text(x, y, '', fontsize=12)
        self.lx.set_ydata(y)
        self.txt.set_position((x, y))
        self.txt.set_text('%.2f' % (y))

    def Clean(self):
        self.plt.cla()
        self.lx = None
        self.txt = None
        self.key_value = None

    def SetTitle(self, title):
        if title != None:
            self.title = unicode(title, "utf-8")

    def SetBaseLine(self, y):
        self.base_line_value = y

    def ShowBaseLine(self):
        if self.base_line_value != None:
            self.base_line = self.plt.axhline(self.base_line_value, c="black", ls="--", lw=1)

    def Refresh(self, data_list, name_list=[]):
        self.Clean()
        self.ax1.set_title(self.title, fontproperties=self.zhfont)
        # self.plt.title(self.title, fontproperties=self.zhfont)
        i = 0
        for iloop in range(len(data_list)):
            x = data_list[iloop][i:i+100, 0]
            y = data_list[iloop][i:i+100, 1]
            if len(name_list) > iloop:
                name = name_list[iloop]
            else:
                name = str(iloop)
            if self.x_is_date:
                xs = [datetime.datetime.strptime(d, '%Y%m%d').date() for d in x.astype(int).astype(str)]
            else:
                xs = x
            self.plt.plot(xs, y, label=name, linewidth=1)
        if self.x_is_date:
            self.plt.gcf().autofmt_xdate()
        self.plt.legend()
        self.plt.show()
        self.ShowBaseLine()
        
        while(1):
            self.plt.pause(0.1)
            if self.key_value != None:
                break
        if self.key_value == 'escape':
            exit()
        return self.key_value

    def RefreshAlign(self, x_data, y_data_list, bar_list=[], name_list=[]):
        self.Clean()
        self.ax1.set_title(self.title, fontproperties=self.zhfont)
        # self.plt.title(self.title, fontproperties=self.zhfont)
        if self.x_is_date:
            xs = [datetime.datetime.strptime(d, '%Y%m%d').date() for d in x_data.astype(int).astype(str)]
        else:
            xs = x_data
        ax2_max = 0
        for iloop in range(len(y_data_list)):
            if len(name_list) > iloop:
                name = name_list[iloop]
            else:
                name = str(iloop)
            show_bar = False
            if len(bar_list) > iloop:
                show_bar = bar_list[iloop]
            if show_bar:
                self.ax2.bar(xs, y_data_list[iloop])
                if ax2_max == 0:
                    ax2_max = max(y_data_list[iloop])
            else:
                self.ax1.plot(xs, y_data_list[iloop], label=name, linewidth=1)

        if ax2_max > 0:
            self.ax2.set_ylim(0, ax2_max * 5)
        if self.x_is_date:
            self.plt.gcf().autofmt_xdate()
        self.plt.legend()
        self.plt.show()
        self.ShowBaseLine()
        
        while(1):
            self.plt.pause(0.1)
            if self.key_value != None:
                break
        if self.key_value == 'escape':
            exit()
        return self.key_value


class Show_():
    def __init__(self, x_is_date = False):
        self.x_is_date = x_is_date
        self.key_value = []
        self.plt, mdate, Cursor, self.lines, self.zhfont = base_common.ImportMatPlot()
        self.plt.ion()
        self.SetTitle('--')
        self.fig = self.plt.figure(dpi=70,figsize=(32,10))
        self.ax1 = self.fig.subplots(1, 1, sharex=True)
        self.ax2 = self.ax1.twinx()

        self.fig.canvas.mpl_connect('key_press_event', lambda event: self.on_key(event))
        self.fig.canvas.mpl_connect('motion_notify_event', lambda event: self.mouse_move(event))
        
        if self.x_is_date:
            self.ax1.xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m-%d'))
        
        self.plt.xlabel('x')
        self.plt.ylabel('y')
        self.plt.grid(True)
        # self.cursor = Cursor(self.ax1, useblit=True, color='black', linewidth=1)
        

    def on_key(self, event):
        self.key_value = event.key

    def mouse_move(self, event):
        if not event.inaxes:
            return
        x, y = event.xdata, event.ydata
        if self.lx == None:
            self.lx = self.plt.axhline(y, c="r", ls="--", lw=1)
            self.txt = self.plt.text(x, y, '', fontsize=12)
        self.lx.set_ydata(y)
        self.txt.set_position((x, y))
        self.txt.set_text('%.2f' % (y))

    def Clean(self):
        self.plt.cla()
        self.lx = None
        self.txt = None
        self.key_value = None

    def SetTitle(self, title):
        if title != None:
            self.title = unicode(title, "utf-8")

    def RefreshAlign(self, x_data, y_data_list, name_list=[]):
        self.Clean()
        self.ax1.set_title(self.title, fontproperties=self.zhfont)
        # self.plt.title(self.title, fontproperties=self.zhfont)
        if self.x_is_date:
            xs = [datetime.datetime.strptime(d, '%Y%m%d').date() for d in x_data.astype(int).astype(str)]
        else:
            xs = x_data
        for iloop in range(len(y_data_list)):
            if len(name_list) > iloop:
                name = name_list[iloop]
            else:
                name = str(iloop)
            ppi = y_data_list[iloop]
            self.ax1.plot(xs, y_data_list[iloop], label=name, linewidth=1)
        if self.x_is_date:
            self.plt.gcf().autofmt_xdate()
        self.plt.legend()
        self.plt.show()
        
        while(1):
            self.plt.pause(0.1)
            if self.key_value != None:
                break
        if self.key_value == 'escape':
            exit()
        return self.key_value


