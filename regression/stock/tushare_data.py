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

#pd.set_option('display.width', 150)  # 设置字符显示宽度
#pd.set_option('display.max_rows', None)  # 设置显示最大行

def StockCodes():
    print(ts.__version__)
    ts.set_token('230c446ae448ec95357d0f7e804ddeebc7a51ff340b4e6e0913ea2fa')
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

    pd.set_option('display.max_rows', 10)  # 设置显示最大行
    #load_df=load_df[load_df['list_date']<=20000101]
    load_df=load_df[load_df['list_date']<=20000101]
    #load_df=load_df[load_df['industry']=='软件服务']
    print(load_df)
    print("\n\n\n")

    code_list=load_df['ts_code'].values

    #code_list=['600872.SH', '000403.SZ', '600729.SH', '600695.SH', '000659.SZ', '600635.SH', '000711.SZ', '600060.SH', '600890.SH', '000617.SZ']
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

def DownloadStocksTrainData():
    print(ts.__version__)
    ts.set_token('230c446ae448ec95357d0f7e804ddeebc7a51ff340b4e6e0913ea2fa')
    pro = ts.pro_api()

    current_date=time.strftime('%Y%m%d',time.localtime(time.time()))
    code_list=StockCodes()
    for code_index in range(0, len(code_list)):
        stock_code=code_list[code_index]
        file_name='./data/'+stock_code+'_'+current_date+'.csv'
        if not os.path.exists(file_name):
            load_df=pro.daily_basic(ts_code=stock_code, start_date='20100101', end_date=current_date)
            load_df.to_csv(file_name)
        print("%-4d : %s 100%%" % (code_index, file_name))

def DownloadStocksPredictData():
    print(ts.__version__)
    ts.set_token('230c446ae448ec95357d0f7e804ddeebc7a51ff340b4e6e0913ea2fa')
    pro = ts.pro_api()

    temp_date_time=datetime.datetime.now()
    current_date=temp_date_time.strftime("%Y%m%d")
    #current_date='20181118'
    temp_date_time=temp_date_time-datetime.timedelta(days=60)
    start_date_str=temp_date_time.strftime("%Y%m%d")
    code_list=StockCodes()
    for code_index in range(0, len(code_list)):
        stock_code=code_list[code_index]
        file_name1='./data/'+stock_code+'_'+current_date+'.csv'
        file_name2='./data/'+stock_code+'_'+start_date_str+"-"+current_date+'.csv'
        if (not os.path.exists(file_name1)) and (not os.path.exists(file_name2)):
            load_df=pro.daily_basic(ts_code=stock_code, start_date=start_date_str, end_date=current_date)
            load_df.to_csv(file_name2)
            time.sleep(1)
        print("%-4d : %s 100%%" % (code_index, stock_code))

def StocksData2TrainData():
    current_date=time.strftime('%Y%m%d',time.localtime(time.time()))
    code_list=StockCodes()
    train_data_list=[]
    for code_index in range(0, len(code_list)):
        stock_code=code_list[code_index]
        file_name='./data/'+stock_code+'_'+current_date+'.csv'
        if os.path.exists(file_name):
            load_df=pd.read_csv(file_name)
            src_df=load_df[['trade_date', 'close', 'turnover_rate']].copy()
            src_df['increase']=0.0
            for iloop in range(0, len(src_df)-1):
                src_df.iloc[iloop,3]=(100*src_df['close'][iloop]/src_df['close'][iloop+1])-100.0
            for day_loop in range(0, len(src_df)-feature_days-1):
                data_unit=[]
                for iloop in range(0, feature_days):
                    data_unit.append(src_df['increase'][day_loop+iloop+1])
                    data_unit.append(src_df['turnover_rate'][day_loop+iloop+1])
                data_unit.append(src_df['increase'][day_loop])
                train_data_list.append(data_unit)
        print("%-4d : %s 100%%" % (code_index, stock_code))
    train_data=np.array(train_data_list)
    order=np.argsort(np.random.random(len(train_data)))
    train_data=train_data[order]
    print("train_data: {}".format(train_data.shape))
    np.save('train_data.npy', train_data)

def StocksData2PredictData():
    temp_date_time=datetime.datetime.now()
    #current_date=temp_date_time.strftime("%Y%m%d")
    current_date='20181118'
    temp_date_time=temp_date_time-datetime.timedelta(days=60)
    start_date_str=temp_date_time.strftime("%Y%m%d")

    code_list=StockCodes()
    predict_data_list=[]
    predict_stock_code_list=[]
    for code_index in range(0, len(code_list)):
        stock_code=code_list[code_index]
        file_name1='./data/'+stock_code+'_'+current_date+'.csv'
        file_name2='./data/'+stock_code+'_'+start_date_str+"-"+current_date+'.csv'
        if os.path.exists(file_name1) or os.path.exists(file_name2):
            if os.path.exists(file_name1):
                load_df=pd.read_csv(file_name1)
            else:
                load_df=pd.read_csv(file_name2)
            if(len(load_df)>(feature_days+1)):
                src_df=load_df[['trade_date', 'close', 'turnover_rate']].copy()
                src_df['increase']=0.0
                for iloop in range(0, feature_days):
                    src_df.iloc[iloop,3]=(100*src_df['close'][iloop]/src_df['close'][iloop+1])-100.0
                day_loop=0
                data_unit=[]
                for iloop in range(0, feature_days):
                    data_unit.append(src_df['increase'][day_loop+iloop])
                    data_unit.append(src_df['turnover_rate'][day_loop+iloop])
                predict_data_list.append(data_unit)
                predict_stock_code_list.append(stock_code)
        print("%-4d : %s 100%%" % (code_index, stock_code))
    predict_data=np.array(predict_data_list)
    predict_stock_code=np.array(predict_stock_code_list)
    print("predict_data: {}".format(predict_data.shape))
    np.save('predict_data.npy', predict_data)
    np.save('predict_stock_code.npy', predict_stock_code)

if __name__ == "__main__":
    temp_stock_codes=StockCodes()
    print("temp_stock_codes:")
    for iloop in range(0,len(temp_stock_codes)):
        print("%-4d : %s" % (iloop, temp_stock_codes[iloop]))
    print("\n\n\n")
