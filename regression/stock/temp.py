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

# print(int(1.9))

# temp_list = tushare_data.PredictTradeDateList()
# print(temp_list)

# arr = np.array([[1.,2.,20190101.],[4.,5.,20190101.],[7.,8.,20190102.],[9.,10.,20190103.]])
# date_list = [20190102., 20190103.]
# print((arr[:, 2] > float('20190101')) & (arr[:, 2] < float('20190103')))
# print((arr[:, 2] > float('20190101')) | (arr[:, 2] == float('20190103')))
# for i in arr:
#     if i[2] in date_list:
#         print(i)
# # print(list(set(arr[:, 2]).intersection(set(date_list))))
# print(arr[(arr[:, 2] > float('20190101')) & (arr[:, 2] < float('20190103'))])
# arr = arr[arr[:, 2] == float('20190101')]
# print(arr)

# def StockCodes(filter_industry):
#     pro = ts.pro_api()

#     file_name = './data/' + 'stock_code' + '.csv'
#     if os.path.exists(file_name):
#         print("read_csv:%s" % file_name)
#         load_df = pd.read_csv(file_name)
#     else:
#         load_df = pro.stock_basic(exchange = '', list_status = 'L', fields = 'ts_code,symbol,name,area,industry,list_date')
#         load_df.to_csv(file_name)

#     load_df = load_df[load_df['list_date'] <= 20090101]
#     load_df = load_df.copy()
#     load_df = load_df.reset_index(drop=True)
    
#     if filter_industry != '':
#         industry_list = filter_industry.split(',')
#         code_valid_list = []
#         for iloop in range(0, len(load_df)):
#             if load_df['industry'][iloop] in industry_list:
#                 code_valid_list.append(True)
#             else:
#                 code_valid_list.append(False)
#         load_df = load_df[code_valid_list]
#     code_list = load_df['ts_code'].values
#     print(load_df)
#     print('StockCodes(%s)[%u]' % (filter_industry, len(code_list)))
#     return code_list

# # code_list = StockCodes('')
# # code_list = StockCodes('银行')
# # code_list = StockCodes('全国地产')
# code_list = StockCodes('软件服务')
# # code_list = StockCodes('保险')
# # code_list = StockCodes('证券')
# code_list = StockCodes('软件服务,证券')
# print(code_list)

# for iloop in range(1,100):
#     # print('\b\b\b\b%4d' % iloop),
#     print('\r%4d' % iloop)
#     time.sleep(0.1)

    
# for i in range(100):
#     sys.stdout.write('\r%s%%' % (i+1))
#     sys.stdout.flush()
#     time.sleep(0.1)

# pp_data = pd.read_csv('./preprocessed_data/preprocess_data_000001.HK_20190215.csv')
# print(pp_data.dtypes)
# print('\n')
# print(pp_data.loc[0,'trade_date'])
# print('\n')


# for day_loop in range(0, len(pp_data)): 
#     # temp_str = pp_data.loc[day_loop,'trade_date']
#     # new_str = temp_str.replace('-', '')
#     # print('temp_str: %s' % new_str)
#     # pp_data.loc[day_loop,'trade_date'] = new_str
#     pp_data.loc[day_loop,'trade_date'] = pp_data.loc[day_loop,'trade_date'].replace('-', '')

# print(pp_data.loc[0,'trade_date'])
# print('\n')
# pp_data['trade_date'] = pp_data['trade_date'].astype(np.int64)
# print(pp_data.dtypes)
# print('\n')
# print(pp_data)
# print('\n')

# print(pp_data.loc[0,'trade_date'])
# print('\n')

def TransferDateType(file_name):
    pp_data = pd.read_csv(file_name)
    for day_loop in range(0, len(pp_data)): 
        pp_data.loc[day_loop,'trade_date'] = pp_data.loc[day_loop,'trade_date'].replace('-', '')
    pp_data['trade_date'] = pp_data['trade_date'].astype(np.int64)
    pp_data.to_csv(file_name)

def TransferDateType(file_name):
    pp_data = pd.read_csv(file_name)
    for day_loop in range(0, len(pp_data)): 
        pp_data.loc[day_loop,'trade_date'] = pp_data.loc[day_loop,'trade_date'].replace('-', '')
    pp_data['trade_date'] = pp_data['trade_date'].astype(np.int64)
    pp_data.to_csv(file_name)

code_list = tushare_data.StockCodes()
for code_index in range(0, len(code_list)):
    stock_code = code_list[code_index]
    stock_pp_file_name = tushare_data.FileNameStockPreprocessedData(stock_code)
    TransferDateType(stock_pp_file_name)
    print('%s 100%%' % stock_pp_file_name)



