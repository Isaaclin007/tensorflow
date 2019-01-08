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

reload(sys)
sys.setdefaultencoding('utf-8')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

# test_data_list=[]
# for iloop in range(0, 3):
#     day_test_data_list=[]
#     test_data_list.append(day_test_data_list)

# for code_index in range(0, 10):
#     for day_loop in range(0,3):
#         data_unit=[]
#         data_unit.append(code_index)
#         data_unit.append(day_loop)
#         test_data_list[day_loop].append(data_unit)
# test_data=np.array(test_data_list)
# print("test_data: {}".format(test_data.shape))
# print(test_data)

# load_df=pd.read_csv("./data/000555.SZ_20181124_train.csv")
# print("load_df")
# print(load_df)

# temp_len=len(load_df)
# df_1=load_df[:temp_len-10]
# print("df_1")
# print(df_1)

# train_data=np.load("./data/000001.SZ_20181124_train.npy")
# print("\ntrain_data: {}".format(train_data.shape))
# print(train_data)
# temp_data=np.load("./data/000002.SZ_20181124_train.npy")
# print("\ntemp_data: {}".format(temp_data.shape))
# print(temp_data)
# train_data=np.vstack((train_data, temp_data))
# print("\ntrain_data: {}".format(train_data.shape))
# print(train_data)

# train_data=np.load("./temp_data/train_data.npy")
# train_data2=np.load("../stock_/train_data.npy")
# if len(train_data)!=len(train_data2):
#     print("len(train_data)!=len(train_data2)")

# for iloop in range(0, len(train_data)):
#     for cloop in range(0,85):
#         # if train_data[iloop, cloop]!=train_data2[iloop, cloop] :
#         #     print("----%u,%u" %(iloop, cloop))
#         if train_data[iloop, 90]!=train_data2[iloop, 86] :
#             print("****%u,%u" %(iloop, cloop))
#     if iloop%100000 == 0 :
#         print("%u" %(iloop))

# def GetTrainData():
#     train_data=np.load("./temp_data/train_data.npy")
#     print("train_data: {}".format(train_data.shape))
#     # raw_input("Enter ...")

#     print("reorder...")
#     order=np.argsort(np.random.random(len(train_data)))
#     train_data=train_data[order]
#     # raw_input("Enter ...")

#     print("get feature and label...")
#     feature_size=tushare_data.FeatureSize()
#     label_index=tushare_data.LabelColIndex()
#     print("get feature ...")
#     train_features=train_data[:,0:feature_size].copy()
#     # raw_input("Enter ...")

#     print("get label...")
#     train_labels=train_data[:,label_index:label_index+1].copy()
#     # raw_input("Enter ...")
#     print("train_features: {}".format(train_features.shape))
#     print("train_labels: {}".format(train_labels.shape))

#     return train_features, train_labels


# f,l=GetTrainData()
# raw_input("Enter ...")

# pro = ts.pro_api()
# df = pro.daily(trade_date='20180810')
# print(df)

# print("\n\n")
# df_basic=pro.daily_basic(trade_date='20180810')
# print(df_basic)

# print("\n\n")
# df=pro.trade_cal(exchange='', start_date='20180101', end_date='20181130')
# print(df)
# print("\n\n")

# tushare_data.DownloadStocksPredictData()

# train_features, train_labels = tushare_data.GetTrainData()
# print("train_features:")
# print(train_features)

# train_data=np.load("./temp_data/train_data.npy")
# print("train_data: {}".format(train_data.shape))
# print(train_data[10])

# load_data=np.load("./temp_data/test_data.npy")
# print("load_data: {}".format(load_data.shape))
# print(load_data[10][6])

# tushare_data.GetTestData()

# temp_str = "837465.SZ"
# print(float(temp_str[0:6]))

# a = 0
# b = 0
# c = 100
# for iloop in reversed(range(0, 10)):

#     b+=10
#     c /= 2
#     print(b)
#     print(c)
#     caption = 'prediction_%d' % iloop
#     print(caption)

print(int(1.9))

