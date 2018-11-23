# -*- coding:UTF-8 -*-

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
from tensorflow import keras
from compiler.ast import flatten
import numpy as np
import pandas as pd
import time
import sys
import tushare_data


# temp_stock_codes=tushare_data.StockCodes()
# print("temp_stock_codes:")
# for iloop in range(0,len(temp_stock_codes)):
#     print("%-4d : %s" % (iloop, temp_stock_codes[iloop]))
# print("\n\n\n")

# tushare_data.DownloadStocksData()
# print("\n\n\n")

# tushare_data.StocksData2TrainData()
# print("\n\n\n")

# tushare_data.DownloadStocksPredictData()
# print("\n\n\n")

tushare_data.DownloadStocksPredictData()
print("\n\n\n")

tushare_data.StocksData2DayTestData()
print("\n\n\n")

# train_data=np.load("train_data.npy")
# print("len(train_data):%d" % len(train_data))
# order=np.argsort(np.random.random(len(train_data)))
# train_data=train_data[order]
# print("reorder finish")

