# -*- coding:UTF-8 -*-

import tensorflow as tf
from tensorflow import keras
import tushare as ts
import numpy as np
import pandas as pd
import os
import time
import sys
import matplotlib.pyplot as plt
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

# def TransferDateType(file_name):
#     pp_data = pd.read_csv(file_name)
#     for day_loop in range(0, len(pp_data)): 
#         pp_data.loc[day_loop,'trade_date'] = pp_data.loc[day_loop,'trade_date'].replace('-', '')
#     pp_data['trade_date'] = pp_data['trade_date'].astype(np.int64)
#     pp_data.to_csv(file_name)

# def TransferDateType(file_name):
#     pp_data = pd.read_csv(file_name)
#     for day_loop in range(0, len(pp_data)): 
#         pp_data.loc[day_loop,'trade_date'] = pp_data.loc[day_loop,'trade_date'].replace('-', '')
#     pp_data['trade_date'] = pp_data['trade_date'].astype(np.int64)
#     pp_data.to_csv(file_name)

# code_list = tushare_data.StockCodes()
# for code_index in range(0, len(code_list)):
#     stock_code = code_list[code_index]
#     stock_pp_file_name = tushare_data.FileNameStockPreprocessedData(stock_code)
#     TransferDateType(stock_pp_file_name)
#     print('%s 100%%' % stock_pp_file_name)

# print(tushare_data.CurrentDate())

# print(sys.argv)


# df = pro.moneyflow(ts_code='002149.SZ', start_date='20190115', end_date='20190315')

# data_set_list = []
# for iloop in range(0,10):
#     data_unit = []
#     for kloop in range(0,10):
#         data_unit.append(iloop * 10 + kloop)
#     data_set_list.append(data_unit)
# data_set = np.array(data_set_list)

# pos1 = data_set[:,5]> 30
# pos2 = data_set[:,5]< 50
# pos3 = pos1 & pos2

# print(pos1)
# print(pos2)
# print(pos3)

# date_list = tushare_data.TradeDateListRange('20170101', '20180201')
# print(date_list)

# a = [1,2,3,4]
# b = [10,20,30,40]

# na = np.array(a)
# nb = np.array(b)
# nc = np.append(na, nb, axis = 0)
# print('nc:')
# print(nc)
# na = na.reshape((4,1))
# nb = nb.reshape((4,1))
# print("na: {}".format(na.shape))

# print('na:')
# print(na)

# print('nb:')
# print(nb)

# nc = np.append(na, nb, axis = 0)
# print('nc:')
# print(nc)

# nd = np.append(na, nb, axis = 1)
# print('nd:')
# print(nd)

# def GetAvg():
#     avg_value = 100.0
#     return True, avg_value

# temp_value = 0.0
# result, temp_value = GetAvg()
# print('temp_value:%f' % temp_value)

# temp_df = pd.read_csv('./breakup_filter_result.csv')
# caption_list = temp_df.columns.values.tolist()
# print(caption_list)
# print('ts_code' in temp_df.columns.values.tolist())
# print('aaa' in temp_df.columns.values.tolist())

# print("\a")

# import numpy as np
# from numpy import random as nr
# r=nr.randint(0,10,size=(5,4))
# print("r:")
# print(r)

# r1 = r.reshape(5,2,2)
# print("r1:")
# print(r1)

# print("train_data: {}".format(r1.shape))

# print(r1.shape[1:])

# data_set = np.load("./data/dataset/dataset_original_14_20120101_20100101_20000101_20190414___2_2_0_1_0_5.npy")
# print("dataset_original: {}".format(data_set.shape))
# print(data_set)

# data_set = np.load("./data/dataset/dataset_14_20120101_20100101_20000101_20190414___2_2_0_1_0_5.npy")
# print("data_set: {}".format(data_set.shape))
# print(data_set)

# x2 = np.arange(10).reshape(2, 5)
# print(x2)
# print("-------------------")
# print(x2[0])
# print("-------------------")
# print(x2[0, :2])
# print("-------------------")
# print(x2[1,2])
# print("-------------------")
# print(x2[1][2])
# print("-------------------")

# temp_list = [[1.0, 8.1], [3.0, 7.3], [2.0, 5.2], [4.0, 1.4], [1.0, 3.1]]
# np_data = np.array(temp_list)

# print(type(temp_list))
# print(type(np_data))

# print('\nnp_data:')
# print(np_data)

# sort_order = np_data[:,0].argsort()
# print('\nsort_order:')
# print(sort_order)

# sort_data = np_data[sort_order]
# print('\nsort_data:')
# print(sort_data)

# sort_order = sort_order[::-1]
# print('\nsort_order:')
# print(sort_order)

# sort_data = np_data[sort_order]
# print('\nsort_data:')
# print(sort_data)

# c0 = np_data[:,0]
# print(c0)
# c0_unique = np.unique(np_data)
# print(c0_unique)
# print(type(c0_unique))

# c0_list = c0_unique.tolist()
# print(c0_list)
# print(type(c0_list))

# for iloop in range(0, 1):
#     trade_list = tushare_data.TradeDateLowLevel('20200414')
#     print('len(trade_list): %u' % len(trade_list))
# print(trade_list)
# trade_list = trade_list.astype(np.int64)
# print(type(trade_list[0]))

# print(tushare_data.TradeDateList('20190414', 10))
# print(tushare_data.TradeDateListRange('20180414', '20190414'))

# tup2 = (1, 2, 3, 4, 5 )
# temp = tup2[2]
# print(temp)
# temp += 1
# print(temp)
# print(tup2)

# import matplotlib.pyplot as plt
# for iloop in range(0, 10):
#     plt.ion()
#     x = np.linspace(-1, iloop, iloop)
    
#     # 绘制普通图像
#     y = x**2
#     plt.cla()
#     plt.figure(dpi=70,figsize=(32,10))
#     plt.plot(x, y, label = "y")
#     plt.plot(x, y, label = "y2")
#     plt.plot(x, y, label = "y3")
#     plt.legend()
#     plt.show()
#     plt.pause(0.001)
#     print(iloop)
#     plt.savefig('./temp.png')

# plt.pause(0.001)

# temp_list = [20190101.0, 20190102.0, 20190103.0, 20190104.0, 20190105.0]
# dataset = np.array(temp_list)
# test_pos = ((dataset.astype(int) % 100) % 4) == 0
# print(type(test_pos))
# print(test_pos)
# print(~test_pos)
# test_data = dataset[test_pos]
# print(test_data)
# train_data = dataset[~test_pos]
# print(train_data)



# plt.ion()
# fig, ax1 = plt.subplots()
# ax2 = ax1.twinx()
# for iloop in range(0, 10):
#     x = np.arange(0,iloop+1,1)
#     y1 = 0.05 * x**2
#     y2 = -1 * y1
#     ax1.cla()
#     ax1.plot(x,y1,'g-')
#     ax2.plot(x,y2,'b-')
    
#     ax1.set_xlabel("X data")
#     ax1.set_ylabel("Y1",color='g')
    
#     ax2.set_ylabel("Y2",color='b')
#     plt.show()
#     plt.pause(1)

# def f():
#     if not hasattr(f, 'x'):
#         f.x = 0
#     print(f.x)
#     f.x+=1

# for iloop in range(0, 10):
#     f()


# plt.ion()
# fig, ax1 = plt.subplots()
# ax2 = ax1.twinx()
# x = np.arange(0,10,1)
# y1 = 0.05 * x**2
# ax1.plot(x,y1,'g-')
# plt.show()
# plt.pause(1)

# def PlotHistory(losses):        
#     np_arr = np.array(losses)
#     if len(np_arr) > 0:
#         # x = np_arr[:,0]
#         # y = np_arr[:,1]
#         print(type(x))
#         print(type(y))
#         print(x)
#         print(y)
        

# a = [[0.0,1.0], [1.0,10.0]]
# PlotHistory(a)

# def Plot2DArray(ax, arr, name):
#     np_arr = np.array(arr)
#     if len(np_arr) > 1:
#         x = np_arr[:,0]
#         y = np_arr[:,1]
#         ax.plot(x, y, label=name)

# def PlotHistory(losses, val_losses, test_increase):
#     if not hasattr(PlotHistory, 'fig'):
#         plt.ion()
#         PlotHistory.fig, PlotHistory.ax1 = plt.subplots()
#         PlotHistory.ax2 = PlotHistory.ax1.twinx()
#     PlotHistory.ax1.cla()
#     PlotHistory.ax2.cla()
#     Plot2DArray(PlotHistory.ax1, losses, 'loss')
#     Plot2DArray(PlotHistory.ax1, val_losses, 'val_loss')
#     Plot2DArray(PlotHistory.ax2, test_increase, 'test_increase')
#     PlotHistory.ax1.legend()
#     PlotHistory.ax2.legend()
#     plt.show()
#     plt.pause(1)
#     # temp_path_name = ModelFilePath(train_mode)
#     # if not os.path.exists(temp_path_name):
#     #     os.makedirs(temp_path_name)
#     # plt.savefig('%s/figure.png' % temp_path_name)

# a = [[0.0,1.0], [1.0,10.0]]
# for iloop in range(0,4):
#     print(iloop)
#     x = np.arange(0,iloop,1)
#     y1 = 0.05 * x**2
#     PlotHistory(a, a, a)

print(np.random.randint(0, 5, size=100))