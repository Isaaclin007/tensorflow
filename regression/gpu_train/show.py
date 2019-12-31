# -*- coding:UTF-8 -*-

import numpy as np
import pandas as pd
import os
import time
import sys
import math
import matplotlib.pyplot as plt


reload(sys)
sys.setdefaultencoding('utf-8')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 


def Plot2DArray(ax, arr, name, color=''):
    np_arr = np.array(arr)
    if len(np_arr) > 1:
        x = np_arr[:,0]
        y = np_arr[:,1]
        if color != '':
            ax.plot(x, y, color, label=name)
        else:
            ax.plot(x, y, label=name)

def PlotHistory(losses, val_losses, test_increase, path_name):
    plt.ion()
    PlotHistory.fig, PlotHistory.ax1 = plt.subplots()
    PlotHistory.ax2 = PlotHistory.ax1.twinx()
    PlotHistory.ax1.cla()
    PlotHistory.ax2.cla()
    Plot2DArray(PlotHistory.ax1, losses, 'loss')
    Plot2DArray(PlotHistory.ax1, val_losses, 'val_loss')
    Plot2DArray(PlotHistory.ax2, test_increase, 'test_increase', 'g-')
    PlotHistory.ax1.legend()
    # plt.show()
    # plt.pause(5)
    if not os.path.exists(path_name):
        os.makedirs(path_name)
    plt.savefig('%s/figure.png' % path_name)
    plt.close()

def ShowHistory(path_name):
    if not os.path.exists(path_name):
        print("ShowHistory.Error: path (%s) not exist" % path_name)
        return
    if os.path.exists('%s/figure.png' % path_name):
        return
    losses = []
    val_losses = []
    test_increase = []

    temp_file_name = '%s/loss.npy' % path_name
    if os.path.exists(temp_file_name):
        losses = np.load(temp_file_name).tolist()
    temp_file_name = '%s/val_losses.npy' % path_name
    if os.path.exists(temp_file_name):
        val_losses = np.load(temp_file_name).tolist()
    temp_file_name = '%s/test_increase.npy' % path_name
    if os.path.exists(temp_file_name):
        test_increase = np.load(temp_file_name).tolist()
        max_increase_epoch = 0.0
        max_increase = 0.0
        for iloop in range(0, len(test_increase)):
            # print("%-8.0f: %.1f" % (test_increase[iloop][0], test_increase[iloop][1]))
            if max_increase < test_increase[iloop][1]:
                max_increase = test_increase[iloop][1]
                max_increase_epoch = test_increase[iloop][0]
        print(path_name)
        # print("----------------------------------------------------")
        print("max increase(%.0f): %.1f" % (max_increase_epoch, max_increase))
        print("")
        captions = ['epoch', 'increase']
        data_df = pd.DataFrame(test_increase, columns=captions)
        data_df.to_csv('%s/test_increase.csv' % path_name)
    if (len(losses) > 0) and (len(val_losses) > 0):
        PlotHistory(losses, val_losses, test_increase, path_name)

def ShowHistoryLevel2(level2_path_name):
    if not os.path.exists(level2_path_name):
        return
    path_list = os.listdir(level2_path_name)
    for path_name in path_list:
        temp_path_name = '%s/%s' % (level2_path_name, path_name)
        if os.path.isdir(temp_path_name):
            ShowHistory(temp_path_name)

if __name__ == "__main__":
    ShowHistoryLevel2('./model/fix')
    ShowHistoryLevel2('./model/wave')
    ShowHistoryLevel2('./model/dqn_fix')


