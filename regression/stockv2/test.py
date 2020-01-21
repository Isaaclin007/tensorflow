# -*- coding:UTF-8 -*-


import tushare as ts
import numpy as np
import pandas as pd
import os
import time
import datetime
import sys
import math
import preprocess
sys.path.append("..")
from common import base_common
from common import np_common

for iloop in range(0):
        print(iloop)
# test_np = np.zeros((5, 9))
# for iloop in range(5):
#     for kloop in range(9):
#         test_np[iloop][kloop] = iloop * 10 + kloop
# shape_list = [5, 3, 3]
# test_np = test_np.reshape(shape_list)
# print(test_np)
# print(test_np[5:20, 0])

# plt, mdate, zhfont = ImportMatPlot()
# fig1 = plt.figure(dpi=70,figsize=(32,10))
# ImportMatPlot()

# stock_pp_file_name = './data/preprocessed/000001.SZ_20200106_0_0_1_f.csv'
# pp_data = pd.read_csv(stock_pp_file_name)

# count = 0
# start_time = time.time()
# test_col_np = np.zeros((len(pp_data), 1))
# for iloop in range(len(pp_data)):
#     a = pp_data.loc[iloop, 'open']
#     b = pp_data.loc[iloop, 'close']
#     test_col_np[iloop][0] = a + b
# new_col = pd.DataFrame(test_col_np, columns=['test'])
# pd.concat([pp_data, new_col], sort=False)
# print(time.time() - start_time)

# np_data = pp_data.values
# # print(np_data)
# start_time = time.time()
# test_col_np = np.zeros((len(np_data), 1))
# for iloop in range(len(np_data)):
#     a = np_data[iloop][3]
#     b = np_data[iloop][4]
#     test_col_np[iloop][0] = a + b
# np_data = np.hstack((np_data, test_col_np))
# print(time.time() - start_time)
# print(np_data)

# test_np = np.zeros((len(pp_data), 2))
# start_time = time.time()
# for iloop in range(len(pp_data)):
#     a = test_np[iloop][0]
#     b = test_np[iloop][1]
# print(time.time() - start_time)


# print(np.random.random(10))

# base_common.MKFileDirs('./a/c/c/c')

# def GetFeature():
#     data_unit = []
#     for iloop in range(150):
#         data_unit.append(iloop)
#     return data_unit

# L = 0
# def SetFeature():
#     # data_unit = []
#     for iloop in range(150):
#         # data_unit.append(iloop)
#         L = iloop
#     # return L

# start = time.time()
# for iloop in range(1000000):
#     SetFeature()
# print(time.time() - start)