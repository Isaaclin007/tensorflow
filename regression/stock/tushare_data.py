# -*- coding:UTF-8 -*-


import tushare as ts
import numpy as np
import pandas as pd
import os
import time
import datetime
import sys
from skimage import feature
from dask.dataframe.methods import size
from llvmlite.ir.types import LabelType

reload(sys)
sys.setdefaultencoding('utf-8')
feature_days = 10
max_predict_day_count = 10  # 决定train_data 和 test_data 的predict_day_count
predict_day_count = 1  # 预测未来几日的数据
referfence_feature_count = 1
test_acture_data_with_feature = False
train_a_stock_min_data_num = 400
train_a_stock_max_data_num = 1000000

# stocks_list_end_date = '20130601'
# train_data_start_date = '20140101'
# train_data_end_date = '20180101'
# test_data_start_date = '20180101'
# test_data_end_date = '20190111'
# train_test_date = '20190111'  # 关联以stock为单位的下载数据
# predict_date = '20190111'

# stocks_list_end_date = '20090101'
# train_data_start_date = '20100101'
# train_data_end_date = '20180601'
# test_data_start_date = '20180601'
# test_data_end_date = '20190111'
# train_test_date = '20190111'
# predict_date = '20190111'

# stocks_list_end_date = '20090101'
# train_data_start_date = '20100101'
# train_data_end_date = '20170101'
# test_data_start_date = '20170101'
# test_data_end_date = '20190111'
# train_test_date = '20190111'
# predict_date = '20181225'

stocks_list_end_date = '20140101'
train_data_start_date = '20140301'
train_data_end_date = '20180101'
test_data_start_date = '20180101'
test_data_end_date = '20190111'
train_test_date = '20190111'
predict_date = '20190127'

# stocks_list_end_date = '20140101'
# train_data_start_date = '20140301'
# train_data_end_date = '20180101'
# test_data_start_date = '20190101'
# test_data_end_date = '20190201'
# train_test_date = '20190201'
# predict_date = '20190127'

code_filter = ''
# industry_filter = '软件服务,互联网,半导体,电脑设备'
# industry_filter = '半导体,电脑设备'
# industry_filter = ''
industry_filter = '软件服务'
# industry_filter = '百货'
# industry_filter = '半导体'
# industry_filter = '保险'
# industry_filter = '玻璃'
# industry_filter = '仓储物流'
# industry_filter = '超市连锁'
# industry_filter = '种植业'
# industry_filter = '出版业'
# industry_filter = '船舶'
# industry_filter = '电脑设备'
# industry_filter = '电器仪表'
# industry_filter = '电气设备'
# industry_filter = '电信运营'
# industry_filter = '多元金融'
# industry_filter = '房产服务'
# industry_filter = '纺织'
# industry_filter = '纺织机械'
# industry_filter = '服饰'
# industry_filter = '钢加工'
# industry_filter = '港口'
# industry_filter = '供气供热'
# industry_filter = '公共交通'
# industry_filter = '公路'
# industry_filter = '工程机械'
# industry_filter = '广告包装'
# industry_filter = '航空'
# industry_filter = '互联网'
# industry_filter = '化工原料'
# industry_filter = '化学制药'
# industry_filter = '环境保护'
# industry_filter = '火力发电'
# industry_filter = '机械基件'
# industry_filter = '家居用品'
# industry_filter = '家用电器'
# industry_filter = '建筑施工'
# industry_filter = '汽车配件'
# industry_filter = '汽车整车'
# industry_filter = '区域地产'
# industry_filter = '全国地产'

feature_size = 87
# feature_size = 22
acture_size = 7
label_col_index = feature_size + predict_day_count - 1

LABEL_PRE_CLOSE_2_TD_CLOSE = 0
LABEL_T1_OPEN_2_TD_CLOSE = 1
LABEL_CONSECUTIVE_RISE_SCORE = 2
LABEL_T1_OPEN_2_TD_OPEN =3
label_type = LABEL_T1_OPEN_2_TD_CLOSE

ts.set_token('230c446ae448ec95357d0f7e804ddeebc7a51ff340b4e6e0913ea2fa')

pd.set_option('display.width', 150)  # 设置字符显示宽度
pd.set_option('display.max_rows', 100)  # 设置显示最大行

def TradeDateList(input_end_date, trade_day_num):
    pro = ts.pro_api()
    # print('TradeDateList(%s, %u)' % (input_end_date, trade_day_num))
    df_trade_cal = pro.trade_cal(exchange = 'SSE', start_date = '20000101', end_date = input_end_date)
    df_trade_cal = df_trade_cal.sort_index(ascending = False)
    df_trade_cal = df_trade_cal[df_trade_cal['is_open'] == 1]
    df_trade_cal = df_trade_cal[:trade_day_num]
    # print(df_trade_cal)
    date_list = df_trade_cal['cal_date'].values
    return date_list

def TradeDateListRange(input_start_date, input_end_date):
    pro = ts.pro_api()

    df_trade_cal = pro.trade_cal(exchange = 'SSE', start_date = input_start_date, end_date = input_end_date)
    df_trade_cal = df_trade_cal.sort_index(ascending = False)
    df_trade_cal = df_trade_cal[df_trade_cal['is_open'] == 1]
    # print(df_trade_cal)
    date_list = df_trade_cal['cal_date'].values
    return date_list

def StockCodeFilter(ts_code, code_filter_list):
    for it in code_filter_list:
        if ts_code[0:len(it)] == it:
            return True
    return False

def StockCodes():
    pro = ts.pro_api()

    file_name = './data/' + 'stock_code' + '.csv'
    if os.path.exists(file_name):
        print("read_csv:%s" % file_name)
        load_df = pd.read_csv(file_name)
    else:
        load_df = pro.stock_basic(exchange = '', list_status = 'L', fields = 'ts_code,symbol,name,area,industry,list_date')
        load_df.to_csv(file_name)

    load_df = load_df[load_df['list_date'] <= int(stocks_list_end_date)]
    load_df = load_df.copy()
    load_df = load_df.reset_index(drop=True)

    industry_filter_en = False
    code_filter_en = False
    if industry_filter != '':
        industry_list = industry_filter.split(',')
        industry_filter_en = True
    if code_filter != '':
        code_filter_list = code_filter.split(',')
        code_filter_en = True

    code_valid_list = []
    for iloop in range(0, len(load_df)):
        temp_code_valid = True
        if industry_filter_en:
            if not load_df['industry'][iloop] in industry_list:
                temp_code_valid = False
        if code_filter_en:
            if not StockCodeFilter(load_df['ts_code'][iloop], code_filter_list):
                temp_code_valid = False
        code_valid_list.append(temp_code_valid)
    load_df = load_df[code_valid_list]
    print(load_df)
    print('StockCodes(%s)[%u]' % (industry_filter, len(load_df)))
    code_list = load_df['ts_code'].values
    return code_list

def FileNameStockDownloadData(stock_code):
    temp_file_name = './download_data/' + stock_code + '_' + train_test_date + '.csv'
    return temp_file_name

def FileNameTradeDayDownloadData(trade_date):
    temp_file_name = './download_data/'+'trade_date'+'_'+trade_date+'.csv'
    return temp_file_name

def DownloadAStocksData(ts_code):
    pro = ts.pro_api()
    start_date = '19000101'
    end_data = train_test_date
    file_name = FileNameStockDownloadData(ts_code)
    if not os.path.exists(file_name):
        df_basic=pro.daily_basic(ts_code = ts_code, start_date = start_date, end_date = end_data)
        df = pro.daily(ts_code = ts_code, start_date = start_date, end_date = end_data)
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
        df_merge = pd.merge(df_basic, df, left_on='trade_date', right_on='trade_date')
        # print("\n\ndf_merge:")
        # print(df_merge.dtypes)
        # print(df_merge)
        df_merge.to_csv(file_name)
    
def DownloadATradeDayData(input_trade_date):
    pro = ts.pro_api()
    file_name = FileNameTradeDayDownloadData(input_trade_date)
    if not os.path.exists(file_name):
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

def LoadATradeDayData(trade_date):
    file_name = FileNameTradeDayDownloadData(trade_date)
    load_df = pd.read_csv(file_name)
    return load_df

def DownloadTrainTestData():
    code_list = StockCodes()
    for code_index in range(0, len(code_list)):
        stock_code = code_list[code_index]
        DownloadAStocksData(stock_code)
        print("%-4d : %s 100%%" % (code_index, stock_code))

def PredictTradeDateList():
    ref_trade_day_num = feature_days + referfence_feature_count + 30 - 1 + 20  # 允许有20天停盘
    date_list = TradeDateList(predict_date, ref_trade_day_num)
    return date_list

def TestTradeDateList():
    date_list = TradeDateListRange(test_data_start_date, test_data_end_date)
    return date_list

def DownloadPredictData():
    date_list = PredictTradeDateList()
    for date_index in range(0, len(date_list)):
        temp_date = date_list[date_index]
        DownloadATradeDayData(temp_date)
        print("%-4d : %s 100%%" % (date_index, temp_date))

def StockDataPreProcess(stock_data_df):
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

def AppendFeature( src_df, feature_day_pointer, data_unit):
    if feature_size == 87:
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
    elif feature_size == 22:
        temp_index = feature_day_pointer
        data_unit.append(src_df['total_share'][temp_index])
        data_unit.append(src_df['float_share'][temp_index])
        for iloop in range(0, feature_days):                
            temp_index=feature_day_pointer+iloop
            data_unit.append(src_df['close_increase'][temp_index])
            data_unit.append(src_df['turnover_rate_f'][temp_index])
        
def AppendLabel( src_df, day_index, data_unit):
    feature_day_pointer = day_index + max_predict_day_count
    if label_type == LABEL_PRE_CLOSE_2_TD_CLOSE:
        feature_last_price = src_df['close'][feature_day_pointer]
        for iloop in reversed(range(0, max_predict_day_count)):
            temp_index = day_index + iloop
            temp_increase_per = ((src_df['close'][temp_index] / feature_last_price) - 1.0) * 100.0
            data_unit.append(temp_increase_per)
    elif label_type == LABEL_T1_OPEN_2_TD_CLOSE:
        feature_last_price = src_df['open'][feature_day_pointer - 1]
        for iloop in reversed(range(0, max_predict_day_count)):
            temp_index = day_index + iloop
            temp_increase_per = ((src_df['close'][temp_index] / feature_last_price) - 1.0) * 100.0
            data_unit.append(temp_increase_per)
    elif label_type == LABEL_T1_OPEN_2_TD_OPEN:
        feature_last_price = src_df['open'][feature_day_pointer - 1]
        for iloop in reversed(range(0, max_predict_day_count)):
            temp_index = day_index + iloop
            temp_increase_per = ((src_df['open'][temp_index] / feature_last_price) - 1.0) * 100.0
            data_unit.append(temp_increase_per)
    elif label_type == LABEL_CONSECUTIVE_RISE_SCORE:
        max_sum_score = -1.0
        sum_score = 0.0
        day_score = 0.0
        for iloop in reversed(range(0, max_predict_day_count)):
            temp_index = day_index + iloop
            # day_score = src_df['close_increase'][temp_index] - 5.0
            if src_df['close_increase'][temp_index] > 0.0:
                day_score = src_df['close_increase'][temp_index]
            else:
                day_score = src_df['close_increase'][temp_index] * 2
            sum_score += day_score
            if sum_score > max_sum_score:
                max_sum_score = sum_score
            data_unit.append(max_sum_score)
    

ACTURE_DATA_INDEX_OPEN_INCREASE = 0
ACTURE_DATA_INDEX_LOW_INCREASE = 1
ACTURE_DATA_INDEX_OPEN = 2
ACTURE_DATA_INDEX_LOW = 3
ACTURE_DATA_INDEX_CLOSE = 4
ACTURE_DATA_INDEX_TSCODE = 5
ACTURE_DATA_INDEX_DATE = 6

def TestDataPredictFeatureOffset(referfence_day_index):
    return (feature_size + acture_size) * referfence_day_index

def TestDataLastPredictFeatureOffset():
    return TestDataPredictFeatureOffset(referfence_feature_count - 1)

def TestDataLastPredictActureOffset():
    return TestDataLastPredictFeatureOffset() + feature_size

def TestDataMonitorDataOffset():
    return (feature_size + acture_size) * referfence_feature_count

def TestDataMonitorFeatureOffset(day_index):
    if test_acture_data_with_feature:
        return TestDataMonitorDataOffset() + (feature_size + acture_size) * day_index
    else:
        return 0

def TestDataMonitorActureOffset(day_index):
    if test_acture_data_with_feature:
        return TestDataMonitorDataOffset() + (feature_size + acture_size) * day_index + feature_size
    else:
        return TestDataMonitorDataOffset() + acture_size * day_index

def TestDataLastMonitorActureOffset():
    return TestDataMonitorActureOffset(predict_day_count - 1)

def AppendActureData(src_df, day_index, data_unit):
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
        AppendFeature(src_df, day_index + max_predict_day_count, data_unit)
        AppendLabel(src_df, day_index, data_unit)

    elif feature_type == FEATURE_TYPE_PREDICT:
        AppendFeature(src_df, day_index, data_unit)
        AppendActureData(src_df, day_index, data_unit)
        
    elif feature_type == FEATURE_TYPE_TEST:
        for iloop in reversed(range(0, referfence_feature_count)):
            predict_day_pointer = day_index + max_predict_day_count + iloop
            AppendFeature(src_df, predict_day_pointer, data_unit)
            AppendActureData(src_df, predict_day_pointer, data_unit)
        for iloop in reversed(range(0, max_predict_day_count)):
            monitor_day_index = day_index + iloop
            if test_acture_data_with_feature:
                AppendFeature(src_df, monitor_day_index, data_unit)
            AppendActureData(src_df, monitor_day_index, data_unit)

    # if day_index == 101:
        # print('\n\n\n')
        # for iloop in range(0, len(data_unit)):
            # print('data_unit[%d]:%f' % (iloop, data_unit[iloop]))
    return data_unit

# 对于整体训练数据，关注
# feature size
# label type
# max_predict_day_count
# 上市日期门限、
# 训练数据起始日期、
# 训练数据截至日期、
# 个股训练数据最小和最大数据量
def FileNameTrainData():
    file_name = './temp_data/train_data_%u_%u_%u_%s_%s_%s_%u_%u_%s%s.npy' % ( \
        feature_size, \
        label_type, \
        max_predict_day_count, \
        stocks_list_end_date, \
        train_data_start_date, \
        train_data_end_date, \
        train_a_stock_min_data_num, \
        train_a_stock_max_data_num, \
        industry_filter, \
        code_filter)
    return file_name

# 对于整体测试数据，关注
# feature size
# label type
# predict_day_count
# acture size
# 测试上市日期门限、
# 测试数据截至日期、
# 测试天数、
# 测试天数采样比例、
# 参考特征天数、
# 测试acture是否包含feature
def FileNameTestData():
    file_name = './temp_data/test_data_%u_%u_%u_%u_%s_%s_%s_%u_%d_%s%s.npy' % ( \
        feature_size, \
        label_type, \
        max_predict_day_count, \
        acture_size, \
        stocks_list_end_date, \
        test_data_start_date, \
        test_data_end_date, \
        referfence_feature_count, \
        int(test_acture_data_with_feature), \
        industry_filter, \
        code_filter)
    return file_name

# 对于PP data，只关注stock code、结束时间
def FileNameStockPreprocessedData(stock_code):
    temp_file_name = './preprocessed_data/preprocess_data_%s_%s.csv' % (\
                    stock_code, \
                    train_test_date)
    return temp_file_name

def TrainStockDataExist(stock_code):
    file_name = FileNameStockDownloadData(stock_code)
    return os.path.exists(file_name)

def TrainStockDataRead(stock_code):
    file_name = FileNameStockDownloadData(stock_code)
    load_df = pd.read_csv(file_name)
    return load_df

def OffsetTradeDate(ref_date, day_offset):
    temp_list = TradeDateList(ref_date, day_offset + 1)
    return temp_list[day_offset]

def UpdateTrainTestData():
    if os.path.exists(FileNameTrainData()) and os.path.exists(FileNameTestData()):
        return
    code_list = StockCodes()
    train_data_init_flag = True
    test_data_init_flag = True
    train_pp_start_date = OffsetTradeDate(train_data_start_date, feature_days + max_predict_day_count)
    train_pp_end_date = train_data_end_date
    test_pp_start_date = OffsetTradeDate(test_data_start_date, feature_days + max_predict_day_count)
    test_pp_end_date = test_data_end_date
    for code_index in range(0, len(code_list)):
        stock_code = code_list[code_index]
        stock_download_file_name = FileNameStockDownloadData(stock_code)
        stock_pp_file_name = FileNameStockPreprocessedData(stock_code)
        if os.path.exists(stock_download_file_name):
            if os.path.exists(stock_pp_file_name):
                pp_data = pd.read_csv(stock_pp_file_name)
            else:
                download_df = pd.read_csv(stock_download_file_name)
                pp_data = StockDataPreProcess(download_df)
                pp_data.to_csv(stock_pp_file_name)
        
            # 拆分 train_pp_data 和 test_pp_data
            train_pp_data = pp_data[ \
                                    (pp_data['trade_date'] > int(train_pp_start_date)) & \
                                    (pp_data['trade_date'] <= int(train_pp_end_date))].copy()
            train_pp_data = train_pp_data.reset_index(drop=True)

            test_pp_data = pp_data[ \
                                    (pp_data['trade_date'] > int(test_pp_start_date)) & \
                                    (pp_data['trade_date'] <= int(test_pp_end_date))].copy()
            test_pp_data = test_pp_data.reset_index(drop=True)

            train_data_list = []
            valid_data_num = len(train_pp_data) - feature_days - max_predict_day_count
            if valid_data_num >= train_a_stock_min_data_num:
                for day_loop in range(0, valid_data_num):
                    data_unit = GetAFeature(train_pp_data, day_loop, FEATURE_TYPE_TRAIN)
                    train_data_list.append(data_unit)
                temp_train_data = np.array(train_data_list)
                if len(temp_train_data) > train_a_stock_max_data_num:
                    order = np.argsort(np.random.random(len(temp_train_data)))
                    temp_train_data = temp_train_data[order]
                    temp_train_data = temp_train_data[:train_a_stock_max_data_num]
                if train_data_init_flag:
                    train_data = temp_train_data
                    train_data_init_flag = False
                else:
                    train_data = np.vstack((train_data, temp_train_data))

            test_data_list = []
            valid_data_num = len(test_pp_data) - feature_days - max_predict_day_count
            if valid_data_num > 0:
                for day_loop in range(0, valid_data_num):
                    data_unit = GetAFeature(test_pp_data, day_loop, FEATURE_TYPE_TEST)
                    test_data_list.append(data_unit)
                temp_np_data = np.array(test_data_list)
                if test_data_init_flag:
                    test_data = temp_np_data
                    test_data_init_flag = False
                else:
                    test_data = np.vstack((test_data, temp_np_data))
            print("%-4d : %s 100%%" % (code_index, stock_code))
            # print("train_data: {}".format(train_data.shape))
            # print(train_data)
    print("train_data: {}".format(train_data.shape))
    np.save(FileNameTrainData(), train_data)
    print("test_data: {}".format(test_data.shape))
    np.save(FileNameTestData(), test_data)

def GetTrainData():
    train_data = np.load(FileNameTrainData())
    print("train_data: {}".format(train_data.shape))
    # raw_input("Enter ...")

    print("reorder...")
    order=np.argsort(np.random.random(len(train_data)))
    train_data=train_data[order]
    train_data=train_data[:2000000]
    # raw_input("Enter ...")

    label_index = label_col_index
    print("get feature ...")
    train_features = train_data[:, 0:feature_size].copy()
    # raw_input("Enter ...")

    print("get label...")
    train_labels = train_data[:, label_index:label_index+1].copy()
    # raw_input("Enter ...")
    print("train_features: {}".format(train_features.shape))
    print("train_labels: {}".format(train_labels.shape))

    return train_features, train_labels

def UpdatePredictData():
    date_list = PredictTradeDateList()
    start_flag = True
    for date_index in range(0, len(date_list)):
        temp_date = date_list[date_index]
        load_df = LoadATradeDayData(temp_date)
        if start_flag:
            merge_df = load_df
            start_flag = False
        else:
            merge_df = merge_df.append(load_df)
        print("%-4d : %s 100%%" % (date_index, temp_date))
    # print(merge_df)
    code_list = StockCodes()
    predict_data_list = []
    for code_index in range(0, len(code_list)):
        stock_code = code_list[code_index]
        stock_df = merge_df[merge_df['ts_code'] == stock_code]
        processed_df = StockDataPreProcess(stock_df)
        if(len(processed_df) >= (feature_days)):
            data_unit = GetAFeature(processed_df, 0, FEATURE_TYPE_PREDICT)
            predict_data_list.append(data_unit)
        else:
            print("------%s:%u" % (stock_code, len(processed_df)))
        print("%-4d : %s 100%%" % (code_index, stock_code))
    predict_data = np.array(predict_data_list)
    print("predict_data: {}".format(predict_data.shape))
    np.save('./temp_data/predict_data.npy', predict_data)

def GetPredictData():
    predict_data=np.load("./temp_data/predict_data.npy")
    print("predict_data: {}".format(predict_data.shape))
    return predict_data

def GetTestData():
    test_data=np.load(FileNameTestData())
    print("test_data: {}".format(test_data.shape))
    # print(test_data)
    print("\n\n\n")

    return test_data

def GetAStockFeatures(ts_code, input_date):
    pro = ts.pro_api()
    temp_file_name = './download_data/' + ts_code + '_' + input_date + '.csv'
    if os.path.exists(temp_file_name):
        df_merge = pd.read_csv(temp_file_name)
    else:
        start_date = '19000101'
        df_basic=pro.daily_basic(ts_code = ts_code, start_date = start_date, end_date = input_date)
        df = pro.daily(ts_code = ts_code, start_date = start_date, end_date = input_date)
        if len(df_basic) != len(df) :
            print("DownloadAStocksData.error.1")
            return
        df.drop(['close','ts_code'],axis=1,inplace=True)
        df_merge = pd.merge(df_basic, df, left_on='trade_date', right_on='trade_date')
        df_merge.to_csv(temp_file_name)

    pp_data = StockDataPreProcess(df_merge)
    test_data_list = []
    for day_loop in range(0, 20):
        data_unit = GetAFeature(pp_data, day_loop, FEATURE_TYPE_PREDICT)
        test_data_list.append(data_unit)
    np_data = np.array(test_data_list)
    return np_data

if __name__ == "__main__":
    temp_stock_codes=StockCodes()
    print("temp_stock_codes:")
    for iloop in range(0,len(temp_stock_codes)):
        print("%-4d : %s" % (iloop, temp_stock_codes[iloop]))
    print("\n\n\n")
