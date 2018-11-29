# -*- coding:UTF-8 -*-


import tushare as ts
import numpy as np
import pandas as pd
import os
import time
import datetime
import sys

reload(sys)
sys.setdefaultencoding('utf-8')
feature_days=10
predict_day_count=5 #预测未来几日的数据
test_day_count=30

ts.set_token('230c446ae448ec95357d0f7e804ddeebc7a51ff340b4e6e0913ea2fa')

pd.set_option('display.width', 150)  # 设置字符显示宽度
pd.set_option('display.max_rows', 100)  # 设置显示最大行

def PredictDate():
    current_date=time.strftime('%Y%m%d',time.localtime(time.time()))
    # current_date='20181124'
    return current_date

def TrainDate():
    return '20181124'

def StockCodes():
    print(ts.__version__)
    pro = ts.pro_api()

    file_name='./data/'+'stock_code'+'.csv'
    if os.path.exists(file_name):
        print("read_csv:%s" % file_name)
        load_df=pd.read_csv(file_name)
    else:
        load_df = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,list_date')
        load_df.to_csv(file_name)

    # print("load_df.dtypes:")
    # print(load_df.dtypes)
    # print("\n\n\n")

    # print(load_df)
    # print("\n\n\n")

    #load_df=load_df[load_df['list_date']<=20000101]
    load_df=load_df[load_df['list_date']<=20090101]
    #load_df=load_df[load_df['industry']=='软件服务']
    print(load_df)
    print("\n\n\n")

    code_list=load_df['ts_code'].values

    # code_list=['600872.SH']
    # print("len(load_df):%d" % len(load_df))
    # for iloop in range(0,len(load_df)):
    #     #code_list.append(load_df['ts_code'][iloop])
    #     print("iloop:%d" % iloop)
    #     temp=load_df['list_date'][iloop]
    #     #print(load_df['list_date'][iloop])
    # print('code_list\n')
    # print(code_list)
    # print("\n\n\n")
    return code_list

def DownloadAStocksData( ts_code, end_data, train ):
    print(ts.__version__)
    ts.set_token('230c446ae448ec95357d0f7e804ddeebc7a51ff340b4e6e0913ea2fa')
    pro = ts.pro_api()
    if train :
        start_date='20100101'
        file_name='./data/'+ts_code+'_'+end_data+'_train.csv'
    else:
        start_date='20180901'
        file_name='./data/'+ts_code+'_'+end_data+'_predict.csv'
    if not os.path.exists(file_name):
        df_basic=pro.daily_basic(ts_code=ts_code, start_date=start_date, end_date=end_data)
        df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_data)
        if len(df_basic) != len(df) :
            print("DownloadAStocksData.error.1")
            return
        # print("\n\ndf_basic:")
        # print(df_basic.dtypes)
        # print(df_basic)
        # print("\n\ndf:")
        # print(df.dtypes)
        # print(df)
        df.drop(['close','ts_code'],axis=1,inplace=True)
        df_merge=pd.merge(df_basic, df, left_on='trade_date', right_on='trade_date')
        # print("\n\ndf_merge:")
        # print(df_merge.dtypes)
        # print(df_merge)
        df_merge.to_csv(file_name)
    


def DownloadStocksTrainData():
    ts.set_token('230c446ae448ec95357d0f7e804ddeebc7a51ff340b4e6e0913ea2fa')
    pro = ts.pro_api()

    code_list=StockCodes()
    for code_index in range(0, len(code_list)):
        stock_code=code_list[code_index]
        DownloadAStocksData(stock_code, TrainDate(), True)
        print("%-4d : %s 100%%" % (code_index, stock_code))

def DownloadStocksPredictData():
    ts.set_token('230c446ae448ec95357d0f7e804ddeebc7a51ff340b4e6e0913ea2fa')
    pro = ts.pro_api()

    code_list=StockCodes()
    for code_index in range(0, len(code_list)):
        stock_code=code_list[code_index]
        DownloadAStocksData(stock_code, PredictDate(), False)
        print("%-4d : %s 100%%" % (code_index, stock_code))

# def StocksData2TrainData():
#     current_date=CurrentDate()
#     code_list=StockCodes()
#     train_data_list=[]
#     for code_index in range(0, len(code_list)):
#         stock_code=code_list[code_index]
#         file_name='./data/'+stock_code+'_'+current_date+'.csv'
#         if os.path.exists(file_name):
#             load_df=pd.read_csv(file_name)
#             src_df=load_df[['trade_date', 'close', 'turnover_rate']].copy()
#             src_df['increase']=0.0
#             for iloop in range(0, len(src_df)-1):
#                 src_df.iloc[iloop,3]=(100*src_df['close'][iloop]/src_df['close'][iloop+1])-100.0
#             for day_loop in range(0, len(src_df)-feature_days-1):
#                 data_unit=[]
#                 for iloop in range(0, feature_days):
#                     data_unit.append(src_df['increase'][day_loop+iloop+1])
#                     data_unit.append(src_df['turnover_rate'][day_loop+iloop+1])
#                 data_unit.append(src_df['increase'][day_loop])
#                 train_data_list.append(data_unit)
#         print("%-4d : %s 100%%" % (code_index, stock_code))
#     train_data=np.array(train_data_list)
#     order=np.argsort(np.random.random(len(train_data)))
#     train_data=train_data[order]
#     print("train_data: {}".format(train_data.shape))
#     np.save('train_data.npy', train_data)

def StockDataPreProcess( stock_data_df ):
    src_df=stock_data_df[
        [
            'total_share', 
            'float_share', 
            'free_share', 
            'total_mv', 
            'circ_mv', 
            'open', 
            'close', 
            'pre_close', 
            'high', 
            'low', 
            'turnover_rate_f'
            ]].copy()
    src_df['open_increase']=0.0
    src_df['close_increase']=0.0
    src_df['high_increase']=0.0
    src_df['low_increase']=0.0
    src_df['close_5_avg']=0.0
    src_df['close_10_avg']=0.0
    src_df['close_30_avg']=0.0
    src_df['close_increase_to_5_avg']=0.0
    src_df['close_increase_to_10_avg']=0.0
    src_df['close_increase_to_30_avg']=0.0
    for day_loop in range(0, (len(src_df)-30)):
        src_df['open_increase'][day_loop]=src_df['open'][day_loop]/src_df['pre_close'][day_loop]*100.0-100.0
        src_df['close_increase'][day_loop]=src_df['close'][day_loop]/src_df['pre_close'][day_loop]*100.0-100.0
        src_df['high_increase'][day_loop]=src_df['high'][day_loop]/src_df['pre_close'][day_loop]*100.0-100.0
        src_df['low_increase'][day_loop]=src_df['low'][day_loop]/src_df['pre_close'][day_loop]*100.0-100.0
        temp_sum=0.0
        for iloop in range(0, 5):
            temp_sum=temp_sum+src_df['close'][day_loop+iloop]
        src_df['close_5_avg'][day_loop]=temp_sum/5
        src_df['close_increase_to_5_avg'][day_loop]=src_df['close'][day_loop]/src_df['close_5_avg'][day_loop]*100.0-100.0

        temp_sum=0.0
        for iloop in range(0, 10):
            temp_sum=temp_sum+src_df['close'][day_loop+iloop]
        src_df['close_10_avg'][day_loop]=temp_sum/10
        src_df['close_increase_to_10_avg'][day_loop]=src_df['close'][day_loop]/src_df['close_10_avg'][day_loop]*100.0-100.0


        temp_sum=0.0
        for iloop in range(0, 30):
            temp_sum=temp_sum+src_df['close'][day_loop+iloop]
        src_df['close_30_avg'][day_loop]=temp_sum/30
        src_df['close_increase_to_30_avg'][day_loop]=src_df['close'][day_loop]/src_df['close_30_avg'][day_loop]*100.0-100.0

    return src_df[:len(src_df)-30]

def FeatureSize():
    return 85

def LabelColIndex():
    return (FeatureSize()+5-1)

def GetAFeature( src_df, day_index, train ):
    data_unit=[]
    feature_day_pointer=day_index
    if train :
        feature_day_pointer=feature_day_pointer+predict_day_count
    temp_index=feature_day_pointer
    data_unit.append(src_df['total_share'][temp_index])
    data_unit.append(src_df['float_share'][temp_index])
    data_unit.append(src_df['free_share'][temp_index])
    data_unit.append(src_df['total_mv'][temp_index])
    data_unit.append(src_df['circ_mv'][temp_index])
    for iloop in range(0, feature_days):
        temp_index=feature_day_pointer+iloop
        data_unit.append(src_df['open_increase'][temp_index])
        data_unit.append(src_df['close_increase'][temp_index])
        data_unit.append(src_df['high_increase'][temp_index])
        data_unit.append(src_df['low_increase'][temp_index])
        data_unit.append(src_df['close_increase_to_5_avg'][temp_index])
        data_unit.append(src_df['close_increase_to_10_avg'][temp_index])
        data_unit.append(src_df['close_increase_to_30_avg'][temp_index])
        data_unit.append(src_df['turnover_rate_f'][temp_index])
    if train :
        temp_increase=1.0
        for iloop in range(0, predict_day_count):
            temp_index=day_index+(predict_day_count-iloop-1)
            temp_increase=temp_increase*((src_df['close_increase'][temp_index]/100.0)+1.0)
            temp_increase_per=temp_increase*100.0-100.0
            data_unit.append(temp_increase_per)

        temp_index=day_index+(predict_day_count-1)
        data_unit.append(src_df['open_increase'][temp_index])
        data_unit.append(src_df['low_increase'][temp_index])
    return data_unit

# def StocksData2TrainTestData():
#     train_date=TrainDate()
#     code_list=StockCodes()
#     train_data_list=[]
#     test_data_list=[]
#     for iloop in range(0, test_day_count):
#         day_test_data_list=[]
#         test_data_list.append(day_test_data_list)
#     for code_index in range(0, len(code_list)):
#         stock_code=code_list[code_index]
#         file_name='./data/'+stock_code+'_'+train_date+'_train.csv'
#         if os.path.exists(file_name):
#             load_df=pd.read_csv(file_name)
#             if len(load_df)>1000:
#                 src_df=StockDataPreProcess(load_df)
#                 # temp_file_name='./data/'+stock_code+'_'+train_date+'_train_preprocess.csv'
#                 # src_df.to_csv(temp_file_name)
#                 for day_loop in range(0, len(src_df)-feature_days-predict_day_count):
#                     data_unit=GetAFeature(src_df, day_loop, True)
#                     if(day_loop<test_day_count):
#                         test_data_list[day_loop].append(data_unit)
#                     else:
#                         train_data_list.append(data_unit)
#         print("%-4d : %s 100%%" % (code_index, stock_code))
#     train_data=np.array(train_data_list)
#     print("train_data: {}".format(train_data.shape))
#     np.save('train_data.npy', train_data)
#     test_data=np.array(test_data_list)
#     print("test_data: {}".format(test_data.shape))
#     np.save('test_data.npy', test_data)

def StocksData2TrainTestData():
    train_date=TrainDate()
    code_list=StockCodes()
    test_data_list=[]
    init_flag=True
    for iloop in range(0, test_day_count):
        day_test_data_list=[]
        test_data_list.append(day_test_data_list)
    for code_index in range(0, len(code_list)):
        stock_code=code_list[code_index]
        file_name='./data/'+stock_code+'_'+train_date+'_train.csv'
        if os.path.exists(file_name):
            load_df=pd.read_csv(file_name)
            if len(load_df)>1000:
                train_data_list=[]
                src_df=StockDataPreProcess(load_df)
                # temp_file_name='./data/'+stock_code+'_'+train_date+'_train_preprocess.csv'
                # src_df.to_csv(temp_file_name)
                for day_loop in range(0, len(src_df)-feature_days-predict_day_count):
                    data_unit=GetAFeature(src_df, day_loop, True)
                    if(day_loop<test_day_count):
                        test_data_list[day_loop].append(data_unit)
                    else:
                        train_data_list.append(data_unit)
                temp_train_data=np.array(train_data_list)
                if init_flag:
                    train_data=temp_train_data
                    init_flag=False
                else:
                    train_data=np.vstack((train_data, temp_train_data))
                print("%-4d : %s 100%%" % (code_index, stock_code))
                # print("train_data: {}".format(train_data.shape))
                # print(train_data)
    print("train_data: {}".format(train_data.shape))
    np.save('./temp_data/train_data.npy', train_data)
    test_data=np.array(test_data_list)
    print("test_data: {}".format(test_data.shape))
    np.save('./temp_data/test_data.npy', test_data)

def GetTrainData():
    train_data=np.load("./temp_data/train_data.npy")
    print("train_data: {}".format(train_data.shape))
    # raw_input("Enter ...")

    print("reorder...")
    order=np.argsort(np.random.random(len(train_data)))
    train_data=train_data[order]
    train_data=train_data[:2000000]
    # raw_input("Enter ...")

    feature_size=FeatureSize()
    label_index=LabelColIndex()
    print("get feature ...")
    train_features=train_data[:,0:feature_size].copy()
    # raw_input("Enter ...")

    print("get label...")
    train_labels=train_data[:,label_index:label_index+1].copy()
    # raw_input("Enter ...")
    print("train_features: {}".format(train_features.shape))
    print("train_labels: {}".format(train_labels.shape))

    return train_features, train_labels

# def StocksData2TrainTestData():
#     train_date=TrainDate()
#     code_list=StockCodes()
#     for code_index in range(0, len(code_list)):
#         stock_code=code_list[code_index]
#         file_name='./data/'+stock_code+'_'+train_date+'_train.csv'
#         file_name_train_data='./data/'+stock_code+'_'+train_date+'_train.npy'
#         if os.path.exists(file_name):
#             load_df=pd.read_csv(file_name)
#             if len(load_df)>1000:
#                 train_data_list=[]
#                 src_df=StockDataPreProcess(load_df)
#                 # temp_file_name='./data/'+stock_code+'_'+train_date+'_train_preprocess.csv'
#                 # src_df.to_csv(temp_file_name)
#                 for day_loop in range(0, len(src_df)-feature_days-predict_day_count):
#                     data_unit=GetAFeature(src_df, day_loop, True)
#                     train_data_list.append(data_unit)
#                 train_data=np.array(train_data_list)
#                 np.save(file_name_train_data, train_data)
#                 print("%-4d : %s 100%%" % (code_index, stock_code))

def StocksData2PredictData():
    predict_date=PredictDate()
    code_list=StockCodes()
    predict_data_list=[]
    predict_stock_code_list=[]
    for code_index in range(0, len(code_list)):
        stock_code=code_list[code_index]
        file_name1='./data/'+stock_code+'_'+predict_date+'_train.csv'
        file_name2='./data/'+stock_code+'_'+predict_date+'_predict.csv'
        if os.path.exists(file_name1) or os.path.exists(file_name2):
            if os.path.exists(file_name1):
                load_df=pd.read_csv(file_name1)
            else:
                load_df=pd.read_csv(file_name2)
            load_df=load_df[:60]
            src_df=StockDataPreProcess(load_df)
            if(len(src_df)>=(feature_days)):
                data_unit=GetAFeature(src_df, 0, False)
                predict_data_list.append(data_unit)
                predict_stock_code_list.append(stock_code)
        print("%-4d : %s 100%%" % (code_index, stock_code))
    predict_data=np.array(predict_data_list)
    predict_stock_code=np.array(predict_stock_code_list)
    print("predict_data: {}".format(predict_data.shape))
    np.save('./temp_data/predict_data.npy', predict_data)
    np.save('./temp_data/predict_stock_code.npy', predict_stock_code)

if __name__ == "__main__":
    temp_stock_codes=StockCodes()
    print("temp_stock_codes:")
    for iloop in range(0,len(temp_stock_codes)):
        print("%-4d : %s" % (iloop, temp_stock_codes[iloop]))
    print("\n\n\n")
