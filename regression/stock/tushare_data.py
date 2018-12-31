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
predict_day_count=10 #预测未来几日的数据
test_day_count=100
referfence_feature_count = 1

ts.set_token('230c446ae448ec95357d0f7e804ddeebc7a51ff340b4e6e0913ea2fa')

pd.set_option('display.width', 150)  # 设置字符显示宽度
pd.set_option('display.max_rows', 100)  # 设置显示最大行


def CurrentDate():
    # current_date=time.strftime('%Y%m%d',time.localtime(time.time()))
    current_date='20181216'
    return current_date

def TrainDate():
    return '20181124'


def TradeDateList():
    pro = ts.pro_api()

    df_trade_cal=pro.trade_cal(exchange='SSE', start_date='20100101', end_date=CurrentDate())
    df_trade_cal=df_trade_cal.sort_index(ascending=False)
    df_trade_cal=df_trade_cal[df_trade_cal['is_open']==1]
    df_trade_cal=df_trade_cal[:feature_days + predict_day_count + test_day_count + 30]
    print(df_trade_cal)
    date_list=df_trade_cal['cal_date'].values
    return date_list

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
    # load_df=load_df[load_df['industry']=='软件服务']
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
    
def DownloadATradeDayData( input_trade_date ):
    pro = ts.pro_api()
    file_name='./data/'+'trade_date'+'_'+input_trade_date+'.csv'
    if not os.path.exists(file_name):
        print(file_name)
        df_basic = pro.daily_basic(trade_date=input_trade_date)
        df = pro.daily(trade_date=input_trade_date)
        if len(df_basic) != len(df) :
            print("DownloadAStocksData.error.1")
            return
        # print("\n\ndf_basic:")
        # print(df_basic.dtypes)
        # print(df_basic)
        # print("\n\ndf:")
        # print(df.dtypes)
        # print(df)
        df.drop(['close','trade_date'],axis=1,inplace=True)
        df_merge=pd.merge(df_basic, df, left_on='ts_code', right_on='ts_code')
        # print("\n\ndf_merge:")
        # print(df_merge.dtypes)
        # print(df_merge)
        df_merge.to_csv(file_name)

def LoadATradeDayData( trade_date ):
    pro = ts.pro_api()
    file_name='./data/'+'trade_date'+'_'+trade_date+'.csv'
    load_df=pd.read_csv(file_name)
    return load_df

def DownloadTrainData():
    pro = ts.pro_api()

    code_list=StockCodes()
    for code_index in range(0, len(code_list)):
        stock_code=code_list[code_index]
        DownloadAStocksData(stock_code, TrainDate(), True)
        print("%-4d : %s 100%%" % (code_index, stock_code))


def DownloadDateData():
    date_list=TradeDateList()
    for date_index in range(0, len(date_list)):
        temp_date=date_list[date_index]
        DownloadATradeDayData(temp_date)
        print("%-4d : %s 100%%" % (date_index, temp_date))

def StockDataPreProcess( stock_data_df ):
    src_df_1=stock_data_df[
        [
            'ts_code',
            'trade_date',
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
            ]]
    src_df_2=src_df_1.copy()
    src_df_2=src_df_2.reset_index(drop=True)
    src_df_2['open_increase']=0.0
    src_df_2['close_increase']=0.0
    src_df_2['high_increase']=0.0
    src_df_2['low_increase']=0.0
    src_df_2['close_5_avg']=0.0
    src_df_2['close_10_avg']=0.0
    src_df_2['close_30_avg']=0.0
    src_df_2['close_increase_to_5_avg']=0.0
    src_df_2['close_increase_to_10_avg']=0.0
    src_df_2['close_increase_to_30_avg']=0.0
    src_df=src_df_2.copy()
    
    avg_5_sum = 0.0
    avg_10_sum = 0.0
    avg_30_sum = 0.0
    avg_5_count = 0
    avg_10_count = 0
    avg_30_count = 0
    for day_loop in reversed(range(0, len(src_df))):
        close_current = src_df.loc[day_loop, 'close']
        # 计算5日均值
        if avg_5_count < 5:
            avg_5_sum += close_current
            avg_5_count += 1
        else:
            avg_5_sum = avg_5_sum + close_current - src_df.loc[day_loop + 5, 'close']
            src_df.loc[day_loop,'close_5_avg'] = avg_5_sum / 5
            
        # 计算10日均值
        if avg_10_count < 10:
            avg_10_sum += close_current
            avg_10_count += 1
        else:
            avg_10_sum = avg_10_sum + close_current - src_df.loc[day_loop + 10, 'close']
            src_df.loc[day_loop,'close_10_avg'] = avg_10_sum / 10
            
        # 计算30日均值
        if avg_30_count < 30:
            avg_30_sum += close_current
            avg_30_count += 1
        else:
            avg_30_sum = avg_30_sum + close_current - src_df.loc[day_loop + 30, 'close']
            src_df.loc[day_loop,'close_30_avg'] = avg_30_sum / 30
            
    temp_open = 0.0
    temp_close = 0.0
    temp_high = 0.0
    temp_low = 0.0
    temp_pre_close = 0.0
    temp_close_5_avg = 0.0
    temp_close_10_avg = 0.0
    temp_close_30_avg = 0.0
    for day_loop in range(0, (len(src_df)-30)):
        temp_open = src_df.loc[day_loop,'open']
        temp_close = src_df.loc[day_loop,'close']
        temp_high = src_df.loc[day_loop,'high']
        temp_low = src_df.loc[day_loop,'low']
        temp_pre_close = src_df.loc[day_loop,'pre_close']
        temp_close_5_avg = src_df.loc[day_loop,'close_5_avg']
        temp_close_10_avg = src_df.loc[day_loop,'close_10_avg']
        temp_close_30_avg = src_df.loc[day_loop,'close_30_avg']
        src_df.loc[day_loop,'open_increase'] = ((temp_open / temp_pre_close) - 1.0) * 100.0
        src_df.loc[day_loop,'close_increase'] = ((temp_close / temp_pre_close) - 1.0) * 100.0
        src_df.loc[day_loop,'high_increase'] = ((temp_high / temp_pre_close) - 1.0) * 100.0
        src_df.loc[day_loop,'low_increase'] = ((temp_low / temp_pre_close) - 1.0) * 100.0
        src_df.loc[day_loop,'close_increase_to_5_avg'] = ((temp_close / temp_close_5_avg) - 1.0) * 100.0
        src_df.loc[day_loop,'close_increase_to_10_avg'] = ((temp_close / temp_close_10_avg) - 1.0) * 100.0
        src_df.loc[day_loop,'close_increase_to_30_avg'] = ((temp_close / temp_close_30_avg) - 1.0) * 100.0

    return src_df[:len(src_df)-30]

def FeatureSize():
    return 87

def PredictDay():
    return predict_day_count

def LabelColIndex():
    return (FeatureSize() + PredictDay() -1)

def TestActureDataIndex():
    return (FeatureSize()*referfence_feature_count + PredictDay())



# train = false:
# features              offset = 0
#     features[5]: pre close
#     features[6]: pre close 5 avg
# + stock_code          offset = feature_size

# train = true:
# features              offset = 0
#     features[5]: pre close
#     features[6]: pre close 5 avg
# + T1_close            offset = feature_size
# + T2_close   
# + ... 
# + Td_close  
# + T1_open_increse     offset = feature_size + predict_day_count + 0
# + T1_low_increase     offset = feature_size + predict_day_count + 1
# + T1_open             offset = feature_size + predict_day_count + 2
# + T1_low              offset = feature_size + predict_day_count + 3
# + Td_close            offset = feature_size + predict_day_count + 4
# + stock_code          offset = feature_size + predict_day_count + 5
# + T1_trade_date       offset = feature_size + predict_day_count + 6

def GetAFeature( src_df, day_index, train, feature_day_count=1 ):
    data_unit=[]
    feature_day_pointer=day_index
    if train :
        feature_day_pointer=feature_day_pointer+predict_day_count
    for feature_day_loop in reversed(range(0, feature_day_count)):
        temp_index = feature_day_pointer + feature_day_loop
        data_unit.append(src_df['total_share'][temp_index])
        data_unit.append(src_df['float_share'][temp_index])
        data_unit.append(src_df['free_share'][temp_index])
        data_unit.append(src_df['total_mv'][temp_index])
        data_unit.append(src_df['circ_mv'][temp_index])
        data_unit.append(src_df['close'][temp_index])
        data_unit.append(src_df['close_5_avg'][temp_index])
        for iloop in range(0, feature_days):
            # if day_index == 101:
                # print('close[%d]: %f' % (iloop, src_df['close'][temp_index]))
                # print('close_5_avg[%d]: %f' % (iloop, src_df['close_5_avg'][temp_index]))
                # print('close_10_avg[%d]: %f' % (iloop, src_df['close_10_avg'][temp_index]))
                # print('close_30_avg[%d]: %f' % (iloop, src_df['close_30_avg'][temp_index]))
                # print('close_increase_to_5_avg[%d]: %f' % (iloop, src_df['close_increase_to_5_avg'][temp_index]))
                # print('close_increase_to_10_avg[%d]: %f' % (iloop, src_df['close_increase_to_10_avg'][temp_index]))
                # print('close_increase_to_30_avg[%d]: %f' % (iloop, src_df['close_increase_to_30_avg'][temp_index]))
                
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
        feature_last_close = src_df['close'][feature_day_pointer]
        # feature_last_close_5_avg = src_df['close_5_avg'][feature_day_pointer]
        for iloop in reversed(range(0, predict_day_count)):
            temp_index = day_index + iloop
            temp_increase_per = ((src_df['close'][temp_index] / feature_last_close) - 1.0) * 100.0
            # temp_increase_per = ((src_df['close_5_avg'][temp_index] / feature_last_close_5_avg) - 1.0) * 100.0
            data_unit.append(temp_increase_per)

        temp_index=day_index+(predict_day_count-1)
        data_unit.append(src_df['open_increase'][temp_index])
        data_unit.append(src_df['low_increase'][temp_index])
        data_unit.append(src_df['open'][temp_index])
        data_unit.append(src_df['low'][temp_index])
        data_unit.append(src_df['close'][day_index])
        temp_str = src_df['ts_code'][temp_index]
        data_unit.append(float(temp_str[0:6]))
        temp_str = src_df['trade_date'][temp_index]
        data_unit.append(float(temp_str))
    else :
        temp_index = feature_day_pointer
        temp_str = src_df['ts_code'][temp_index]
        data_unit.append(float(temp_str[0:6]))
        
    # if day_index == 101:
        # print('\n\n\n')
        # for iloop in range(0, len(data_unit)):
            # print('data_unit[%d]:%f' % (iloop, data_unit[iloop]))
    return data_unit

def UpdateTrainData():
    train_date=TrainDate()
    code_list=StockCodes()
    init_flag=True
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
                    if(day_loop >= test_day_count):
                        data_unit=GetAFeature(src_df, day_loop, True)
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

def UpdatePredictData():
    date_list=TradeDateList()
    start_flag=True
    for date_index in range(0, feature_days+30):
        temp_date=date_list[date_index]
        load_df=LoadATradeDayData(temp_date)
        if start_flag:
            merge_df=load_df
            start_flag=False
        else:
            merge_df=merge_df.append(load_df)
        print("%-4d : %s 100%%" % (date_index, temp_date))
    # print(merge_df)
    code_list=StockCodes()
    predict_data_list=[]
    for code_index in range(0, len(code_list)):
        stock_code=code_list[code_index]
        stock_df=merge_df[merge_df['ts_code']==stock_code]
        processed_df=StockDataPreProcess(stock_df)
        if(len(processed_df)>=(feature_days)):
            data_unit=GetAFeature(processed_df, 0, False)
            predict_data_list.append(data_unit)
        else:
            print("------%s:%u" % (stock_code, len(processed_df)))
        print("%-4d : %s 100%%" % (code_index, stock_code))
    predict_data=np.array(predict_data_list)
    print("predict_data: {}".format(predict_data.shape))
    np.save('./temp_data/predict_data.npy', predict_data)

def GetPredictData():
    predict_data=np.load("./temp_data/predict_data.npy")
    print("predict_data: {}".format(predict_data.shape))
    return predict_data

def UpdateTestData():
    date_list=TradeDateList()
    start_flag=True
    for date_index in range(0, len(date_list)):
        temp_date=date_list[date_index]
        load_df=LoadATradeDayData(temp_date)
        if start_flag:
            merge_df=load_df
            start_flag=False
        else:
            merge_df=merge_df.append(load_df)
        print("%-4d : %s 100%%" % (date_index, temp_date))
    # print(merge_df)
    code_list=StockCodes()
    test_data_list=[]
    for iloop in range(0, test_day_count):
        day_test_data_list=[]
        test_data_list.append(day_test_data_list)
    test_stock_code_list=[]
    for code_index in range(0, len(code_list)):
        stock_code=code_list[code_index]
        stock_df=merge_df[merge_df['ts_code']==stock_code]
        processed_df=StockDataPreProcess(stock_df)
        if (len(processed_df)-feature_days-predict_day_count) >= test_day_count :
            for day_loop in range(0, test_day_count) :
                data_unit=GetAFeature(processed_df, day_loop, True, referfence_feature_count)
                test_data_list[day_loop].append(data_unit)
        print("%-4d : %s 100%%" % (code_index, stock_code))
    test_data=np.array(test_data_list)
    test_stock_code=np.array(code_list)
    print("test_data: {}".format(test_data.shape))
    np.save('./temp_data/test_data_trade_date.npy', test_data)

def GetTestData():
    test_data=np.load("./temp_data/test_data_trade_date.npy")
    print("test_data: {}".format(test_data.shape))
    # print(test_data)
    print("\n\n\n")

    return test_data

if __name__ == "__main__":
    temp_stock_codes=StockCodes()
    print("temp_stock_codes:")
    for iloop in range(0,len(temp_stock_codes)):
        print("%-4d : %s" % (iloop, temp_stock_codes[iloop]))
    print("\n\n\n")
