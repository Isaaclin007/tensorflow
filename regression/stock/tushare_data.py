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
predict_day_count = 10 #预测未来几日的数据
test_day_count = 100
test_day_sample = 1  # test_day_count采样比例
referfence_feature_count = 1
test_acture_data_with_feature = False
# train_a_stock_min_data_num = 100
# train_a_stock_max_data_num = 400
train_a_stock_min_data_num = 1000
train_a_stock_max_data_num = 1000000

ts.set_token('230c446ae448ec95357d0f7e804ddeebc7a51ff340b4e6e0913ea2fa')

pd.set_option('display.width', 150)  # 设置字符显示宽度
pd.set_option('display.max_rows', 100)  # 设置显示最大行


def PredictDate():
    # current_date=time.strftime('%Y%m%d',time.localtime(time.time()))
    current_date='20190104'
    return current_date

def TrainDate():
    return '20181124'

def TestDate():
    return '20190104'

def PredictStockCodeListDate():
    return 20180101

def TestStockCodeListDate():
    # return 20180101
    return 20090101

def TrainStockCodeListDate():
    # return 20180101
    # return 20160101
    return 20090101

def TrainStockStartDate():
    return '20100101'
    # return '20160601'

def TradeDateList(input_end_data):
    pro = ts.pro_api()

    df_trade_cal=pro.trade_cal(exchange = 'SSE', start_date = '20100101', end_date = input_end_data)
    df_trade_cal=df_trade_cal.sort_index(ascending=False)
    df_trade_cal=df_trade_cal[df_trade_cal['is_open']==1]
    df_trade_cal=df_trade_cal[:feature_days + predict_day_count + test_day_count + referfence_feature_count + 30]
    print(df_trade_cal)
    date_list=df_trade_cal['cal_date'].values
    return date_list

def StockCodes(input_list_date):
    print(ts.__version__)
    pro = ts.pro_api()

    file_name='./data/'+'stock_code'+'.csv'
    if os.path.exists(file_name):
        print("read_csv:%s" % file_name)
        load_df=pd.read_csv(file_name)
    else:
        load_df = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,list_date')
        load_df.to_csv(file_name)

    load_df=load_df[load_df['list_date'] <= input_list_date]
    # load_df=load_df[load_df['industry']=='软件服务']

    code_list=load_df['ts_code'].values
    return code_list

def DownloadAStocksData( ts_code, end_data):
    pro = ts.pro_api()
    start_date = TrainStockStartDate()
    file_name = './data/' + ts_code + '_' + end_data + '_train.csv'
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
            print("DownloadATradeDayData.error.1")
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

    code_list = StockCodes(TrainStockCodeListDate())
    for code_index in range(0, len(code_list)):
        stock_code=code_list[code_index]
        DownloadAStocksData(stock_code, TrainDate())
        print("%-4d : %s 100%%" % (code_index, stock_code))


def DownloadPredictData():
    date_list=TradeDateList(PredictDate())
    for date_index in range(0, len(date_list)):
        temp_date=date_list[date_index]
        DownloadATradeDayData(temp_date)
        print("%-4d : %s 100%%" % (date_index, temp_date))

def DownloadTestData():
    date_list=TradeDateList(TestDate())
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
    
def ActureSize():
    return 7

def PredictDay():
    return predict_day_count

def LabelColIndex():
    return (FeatureSize() + PredictDay() -1)




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
def AppendFeature( src_df, feature_day_pointer, data_unit):
    temp_index = feature_day_pointer
    data_unit.append(src_df['total_share'][temp_index])
    data_unit.append(src_df['float_share'][temp_index])
    data_unit.append(src_df['free_share'][temp_index])
    data_unit.append(src_df['total_mv'][temp_index])
    data_unit.append(src_df['circ_mv'][temp_index])
    data_unit.append(src_df['close'][temp_index])
    data_unit.append(src_df['close_5_avg'][temp_index])
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
        
def AppendLabel( src_df, day_index, data_unit):
    feature_day_pointer = day_index + predict_day_count
    feature_last_close = src_df['close'][feature_day_pointer]
    for iloop in reversed(range(0, predict_day_count)):
        temp_index = day_index + iloop
        temp_increase_per = ((src_df['close'][temp_index] / feature_last_close) - 1.0) * 100.0
        data_unit.append(temp_increase_per)

ACTURE_DATA_INDEX_OPEN_INCREASE = 0
ACTURE_DATA_INDEX_LOW_INCREASE = 1
ACTURE_DATA_INDEX_OPEN = 2
ACTURE_DATA_INDEX_LOW = 3
ACTURE_DATA_INDEX_CLOSE = 4
ACTURE_DATA_INDEX_TSCODE = 5
ACTURE_DATA_INDEX_DATE = 6

def TestDataPredictFeatureOffset(referfence_day_index):
    return (FeatureSize() + ActureSize()) * referfence_day_index

def TestDataLastPredictFeatureOffset():
    return TestDataPredictFeatureOffset(referfence_feature_count - 1)

def TestDataLastPredictActureOffset():
    return TestDataLastPredictFeatureOffset() + FeatureSize()

def TestDataMonitorDataOffset():
    return (FeatureSize() + ActureSize()) * referfence_feature_count

def TestDataMonitorFeatureOffset(day_index):
    if test_acture_data_with_feature:
        return TestDataMonitorDataOffset() + (FeatureSize() + ActureSize()) * day_index
    else:
        return 0

def TestDataMonitorActureOffset(day_index):
    if test_acture_data_with_feature:
        return TestDataMonitorDataOffset() + (FeatureSize() + ActureSize()) * day_index + FeatureSize()
    else:
        return TestDataMonitorDataOffset() + ActureSize() * day_index

def TestDataLastMonitorActureOffset():
    return TestDataMonitorActureOffset(predict_day_count - 1)

def AppendActureData( src_df, day_index, data_unit):
    temp_index = day_index
    data_unit.append(src_df['open_increase'][temp_index])
    data_unit.append(src_df['low_increase'][temp_index])
    data_unit.append(src_df['open'][temp_index])
    data_unit.append(src_df['low'][temp_index])
    data_unit.append(src_df['close'][temp_index])
    temp_str = src_df['ts_code'][temp_index]
    data_unit.append(float(temp_str[0:6]))
    temp_str = src_df['trade_date'][temp_index]
    data_unit.append(float(temp_str))
        
FEATURE_TYPE_TRAIN = 0
FEATURE_TYPE_PREDICT = 1
FEATURE_TYPE_TEST = 2
def GetAFeature( src_df, day_index, feature_type):
    data_unit=[]
    if feature_type == FEATURE_TYPE_TRAIN:
        AppendFeature(src_df, day_index + predict_day_count, data_unit)
        AppendLabel(src_df, day_index, data_unit)

    elif feature_type == FEATURE_TYPE_PREDICT:
        AppendFeature(src_df, day_index, data_unit)
        AppendActureData(src_df, day_index, data_unit)
        
    elif feature_type == FEATURE_TYPE_TEST:
        for iloop in reversed(range(0, referfence_feature_count)):
            predict_day_pointer = day_index + predict_day_count + iloop
            AppendFeature(src_df, predict_day_pointer, data_unit)
            AppendActureData(src_df, predict_day_pointer, data_unit)
        for iloop in reversed(range(0, predict_day_count)):
            monitor_day_index = day_index + iloop
            if test_acture_data_with_feature:
                AppendFeature(src_df, monitor_day_index, data_unit)
            AppendActureData(src_df, monitor_day_index, data_unit)

    # if day_index == 101:
        # print('\n\n\n')
        # for iloop in range(0, len(data_unit)):
            # print('data_unit[%d]:%f' % (iloop, data_unit[iloop]))
    return data_unit

def TrainDataFileName():
    file_name = './temp_data/train_data_%s_%s_%s_%u_%u.npy' \
        % (TrainStockCodeListDate(), \
        TrainStockStartDate(), \
        TrainDate(), \
        train_a_stock_min_data_num, \
        train_a_stock_max_data_num)
    return file_name

def TrainStockDataExist(stock_code):
    file_name = './data/' + stock_code + '_' + TrainDate() + '_train.csv'
    return os.path.exists(file_name)

def GetAStockData(stock_code):
    file_name = './data/' + stock_code + '_' + TrainDate() + '_train.csv'
    load_df = pd.read_csv(file_name)
    load_df = load_df[load_df['trade_date'] >= int(TrainStockStartDate())]
    return load_df

def UpdateTrainData():
    code_list=StockCodes(TrainStockCodeListDate())
    init_flag=True
    for code_index in range(0, len(code_list)):
        stock_code = code_list[code_index]
        if TrainStockDataExist(stock_code):
            train_data_list=[]
            temp_file_name = './preprocessed_data/train_preprocess_data_%s_%s.csv' % (stock_code, TrainDate())
            if os.path.exists(temp_file_name):
                src_df = pd.read_csv(temp_file_name)
            else:
                load_df = GetAStockData(stock_code)
                src_df = StockDataPreProcess(load_df)
                src_df.to_csv(temp_file_name)
            valid_data_num = len(src_df) - feature_days-predict_day_count
            if valid_data_num >= (train_a_stock_min_data_num + test_day_count):
                for day_loop in range(0, valid_data_num):
                    if(day_loop >= test_day_count):
                        data_unit = GetAFeature(src_df, day_loop, FEATURE_TYPE_TRAIN)
                        train_data_list.append(data_unit)
                temp_train_data = np.array(train_data_list)
                if len(temp_train_data) > train_a_stock_max_data_num:
                    order = np.argsort(np.random.random(len(temp_train_data)))
                    temp_train_data = temp_train_data[order]
                    temp_train_data = temp_train_data[:train_a_stock_max_data_num]
                if init_flag:
                    train_data = temp_train_data
                    init_flag = False
                else:
                    train_data = np.vstack((train_data, temp_train_data))
                print("%-4d : %s 100%%" % (code_index, stock_code))
                # print("train_data: {}".format(train_data.shape))
                # print(train_data)
    print("train_data: {}".format(train_data.shape))
    np.save(TrainDataFileName(), train_data)

def GetTrainData():
    train_data=np.load(TrainDataFileName())
    print("train_data: {}".format(train_data.shape))
    # raw_input("Enter ...")

    print("reorder...")
    order=np.argsort(np.random.random(len(train_data)))
    train_data=train_data[order]
    train_data=train_data[:1000000]
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
    date_list=TradeDateList(PredictDate())
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
    code_list=StockCodes(PredictStockCodeListDate())
    predict_data_list=[]
    for code_index in range(0, len(code_list)):
        stock_code=code_list[code_index]
        stock_df=merge_df[merge_df['ts_code']==stock_code]
        processed_df=StockDataPreProcess(stock_df)
        if(len(processed_df)>=(feature_days)):
            data_unit=GetAFeature(processed_df, 0, FEATURE_TYPE_PREDICT)
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

def TestDataFileName():
    file_name = './temp_data/test_data_%s_%s_%u_%u_%u_%d.npy' \
        % (TestStockCodeListDate(), \
        TestDate(), \
        test_day_count, \
        test_day_sample, \
        referfence_feature_count, \
        int(test_acture_data_with_feature))
    return file_name

def UpdateTestData():
    date_list=TradeDateList(TestDate())
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
    code_list=StockCodes(TestStockCodeListDate())
    test_data_list=[]
    for iloop in range(0, test_day_count/test_day_sample):
        day_test_data_list=[]
        test_data_list.append(day_test_data_list)
    for code_index in range(0, len(code_list)):
        stock_code=code_list[code_index]
        temp_file_name='./preprocessed_data/test_preprocess_data_%s_%s.csv' % (stock_code, TestDate())
        if os.path.exists(temp_file_name):
            processed_df = pd.read_csv(temp_file_name)
        else:
            stock_df = merge_df[merge_df['ts_code'] == stock_code]
            processed_df = StockDataPreProcess(stock_df)
            processed_df.to_csv(temp_file_name)
        if (len(processed_df) - feature_days - predict_day_count - referfence_feature_count) >= test_day_count :
            for day_loop in range(0, test_day_count) :
                if (day_loop % test_day_sample) == 0:
                    data_unit=GetAFeature(processed_df, day_loop, FEATURE_TYPE_TEST)
                    test_data_list[day_loop/test_day_sample].append(data_unit)
        print("%-4d : %s 100%%" % (code_index, stock_code))
    test_data=np.array(test_data_list)
    print("test_data: {}".format(test_data.shape))
    np.save(TestDataFileName(), test_data)

def GetTestData():
    test_data=np.load(TestDataFileName())
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
