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
import hk_2_tu_data
import math
import preprocess

pro = ts.pro_api()

preprocess_ref_days = 100

reload(sys)
sys.setdefaultencoding('utf-8')
max_predict_day_count = 10  # 决定train_data 和 test_data 的predict_day_count
predict_day_count = 2  # 预测未来几日的数据
referfence_feature_count = 1
test_acture_data_with_feature = False
train_a_stock_min_data_num = 400
train_a_stock_max_data_num = 1000000

use_daily_basic = False
use_money_flow = False
use_adj_factor = True
use_turnover_rate_f = False

DATA_TYPE_DAY = 0
DATA_TYPE_WEEK = 1
data_type = DATA_TYPE_DAY

# FEATURE_G7_10D8 = 0
# FEATURE_G2_10D2 = 1
# FEATURE_G0_10D2 = 2
# FEATURE_G2_10D8 = 3
# FEATURE_G7_10AVG102_10D8 = 4
# FEATURE_G0_10D5 = 5
# FEATURE_G0_10D5_TO_30_AVG = 6
# FEATURE_G0_10D5_TO_100_AVG = 7
# FEATURE_G0_10W5_TO_100_AVG = 8
# FEATURE_G0_100D5_TO_100_AVG = 9
# FEATURE_G0_100D2_TO_100_AVG = 10
# FEATURE_G0_10D14_TO_100_AVG = 11
# FEATURE_G0_10D10 = 12
# FEATURE_G0_10D11_AVG = 13
# FEATURE_G0_10D11_AVG_WITH_DATE = 14
# FEATURE_G0_30D5_AVG = 15
# FEATURE_G0_30D11_AVG = 16
# FEATURE_G0_30D17_AVG = 17

# feature_type = FEATURE_G0_30D5_AVG
# if feature_type == FEATURE_G7_10D8:
#     feature_size = 7 + (8 * 10)
#     feature_relate_days = 10
#     use_daily_basic = True
#     use_money_flow = False
# elif feature_type == FEATURE_G2_10D2:
#     feature_size = 2 + (2 * 10)
#     feature_relate_days = 10
#     use_daily_basic = True
#     use_money_flow = False
# elif feature_type == FEATURE_G0_10D2:
#     feature_size = 0 + (2 * 10)
#     feature_relate_days = 10
#     use_daily_basic = True
#     use_money_flow = False
# elif feature_type == FEATURE_G2_10D8:
#     feature_size = 2 + (8 * 10)
#     feature_relate_days = 10
#     use_daily_basic = True
#     use_money_flow = False
# elif feature_type == FEATURE_G7_10AVG102_10D8:
#     feature_size = 7 + (2 * 10) + (8 * 10)
#     feature_relate_days = 100
#     use_daily_basic = True
#     use_money_flow = False
# elif feature_type == FEATURE_G0_10D5:
#     feature_size = (5 * 10)
#     feature_relate_days = 10
#     use_daily_basic = True
#     use_money_flow = False
# elif feature_type == FEATURE_G0_10D5_TO_30_AVG:
#     feature_size = (5 * 10)
#     feature_relate_days = 10
#     use_daily_basic = True
#     use_money_flow = False
# elif feature_type == FEATURE_G0_10D5_TO_100_AVG:
#     feature_size = (5 * 10)
#     feature_relate_days = 10
#     use_daily_basic = False
#     use_money_flow = False
# elif feature_type == FEATURE_G0_10W5_TO_100_AVG:
#     feature_size = (5 * 10)
#     feature_relate_days = 5 * 10
#     use_daily_basic = False
#     use_money_flow = False
# elif feature_type == FEATURE_G0_100D5_TO_100_AVG:
#     feature_size = (5 * 100)
#     feature_relate_days = 100
#     use_daily_basic = False
#     use_money_flow = False
# elif feature_type == FEATURE_G0_100D2_TO_100_AVG:
#     feature_size = (2 * 100)
#     feature_relate_days = 100
#     use_daily_basic = False
#     use_money_flow = False
# elif feature_type == FEATURE_G0_10D14_TO_100_AVG:
#     feature_size = (14 * 10)
#     feature_relate_days = 100
#     use_daily_basic = False
#     use_money_flow = True
# elif feature_type == FEATURE_G0_10D10:
#     feature_size = (10 * 10)
#     feature_relate_days = 10
#     use_daily_basic = False
#     use_money_flow = False
# elif feature_type == FEATURE_G0_10D11_AVG:
#     feature_relate_days = 10
#     feature_size_one_day = 11
#     feature_size = feature_size_one_day * feature_relate_days
#     use_daily_basic = False
#     use_money_flow = False
# elif feature_type == FEATURE_G0_10D11_AVG_WITH_DATE:
#     feature_relate_days = 10
#     feature_size_one_day = 12
#     feature_size = feature_size_one_day * feature_relate_days
#     use_daily_basic = False
#     use_money_flow = False
# elif feature_type == FEATURE_G0_30D5_AVG:
#     feature_relate_days = 30
#     feature_size_one_day = 5
#     feature_size = feature_size_one_day * feature_relate_days
#     use_daily_basic = False
#     use_money_flow = False
# elif feature_type == FEATURE_G0_30D11_AVG:
#     feature_relate_days = 30
#     feature_size_one_day = 11
#     feature_size = feature_size_one_day * feature_relate_days
#     use_daily_basic = False
#     use_money_flow = False
# elif feature_type == FEATURE_G0_30D17_AVG:
#     feature_relate_days = 30
#     feature_size_one_day = 17
#     feature_size = feature_size_one_day * feature_relate_days
#     use_daily_basic = False
#     use_money_flow = False

# LABEL_PRE_CLOSE_2_TD_CLOSE = 0
# LABEL_T1_OPEN_2_TD_CLOSE = 1
# LABEL_CONSECUTIVE_RISE_SCORE = 2
# LABEL_T1_OPEN_2_TD_OPEN =3
# label_type = LABEL_T1_OPEN_2_TD_CLOSE

# acture_size = 7
# label_col_index = feature_size + predict_day_count - 1

def CurrentDate():
    return time.strftime('%Y%m%d',time.localtime(time.time()))

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

pp_data_start_date = '20000101'
stocks_list_end_date = '20000101'
train_data_start_date = '20120101'
train_data_end_date = '20170101'
test_data_start_date = '20170101'
test_data_end_date = '20190414'
train_test_date = '20190414'
# train_test_date = '20190513'
# train_test_date = CurrentDate()
predict_date = '20181225'

# stocks_list_end_date = '20140101'
# train_data_start_date = '20140301'
# train_data_end_date = '20180101'
# test_data_start_date = '20180101'
# test_data_end_date = '20190111'
# train_test_date = '20190111'
# predict_date = '20190127'

# stocks_list_end_date = '20140101'
# train_data_start_date = '20140301'
# train_data_end_date = '20180101'
# test_data_start_date = '20190101'
# test_data_end_date = '20190201'
# train_test_date = '20190201'
# predict_date = '20190127'

code_filter = ''

# 软件服务 wave_test 20180101前测试数据前十
# code_filter = '000938.SZ,600446.SH,600570.SH,000662.SZ,002195.SZ,600556.SH,000555.SZ,600536.SH,600571.SH'
# wave_test 20180101前测试数据前20
# code_filter = '600701.SH,\
# 600053.SH,\
# 000025.SZ,\
# 000938.SZ,\
# 000584.SZ,\
# 600862.SH,\
# 000856.SZ,\
# 002071.SZ,\
# 600446.SH,\
# 600179.SH,\
# 600745.SH,\
# 600822.SH,\
# 000676.SZ,\
# 002075.SZ,\
# 600055.SH,\
# 002113.SZ,\
# 600570.SH,\
# 000796.SZ,\
# 000913.SZ,\
# 000559.SZ'

# code_filter = '600556.SH,002184.SZ,002232.SZ'


# code_filter = '000001.SH,002415,000650,000937,600104'
# industry_filter = '软件服务,互联网,半导体,电脑设备,百货,仓储物流,电脑设备,电器设备'
# industry_filter = '半导体,电脑设备'
industry_filter = ''
# industry_filter = 'hk'
# industry_filter = '软件服务'
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

min_circ_mv = 0
max_circ_mv = 0


ts.set_token('230c446ae448ec95357d0f7e804ddeebc7a51ff340b4e6e0913ea2fa')

pd.set_option('display.width', 150)  # 设置字符显示宽度
pd.set_option('display.max_rows', 20)  # 设置显示最大行

def TradeDateLowLevel(input_end_date): 
    file_name = './data/trade_date/trade_date_%s.npy' % input_end_date
    if os.path.exists(file_name):
        date_np = np.load(file_name)
        return date_np
    else:
        df_trade_cal = pro.trade_cal(exchange = 'SSE', start_date = '19800101', end_date = input_end_date)
        df_trade_cal.to_csv(file_name)
        df_trade_cal = df_trade_cal.sort_index(ascending = False)
        df_trade_cal = df_trade_cal[df_trade_cal['is_open'] == 1]
        date_np = df_trade_cal['cal_date'].values
        np.save(file_name, date_np)
    return date_np

def TradeDateList(input_end_date, trade_day_num):
    date_np = TradeDateLowLevel(input_end_date)
    return date_np[:trade_day_num].copy()

def TradeDateListRange(input_start_date, input_end_date):
    date_np = TradeDateLowLevel(input_end_date)
    pos = date_np.astype(np.int64) >= int(input_start_date)
    return date_np[pos].copy()

def StockCodeFilter(ts_code, code_filter_list):
    for it in code_filter_list:
        if ts_code[0:len(it)] == it:
            return True
    return False

def StockCodesName(input_stocks_list_end_date, input_industry_filter, input_code_filter):
    pro = ts.pro_api()

    if input_industry_filter == 'hk':
        return hk_2_tu_data.HKCodeList()
    else:
        file_name = './data/' + 'stock_code' + '.csv'
        if os.path.exists(file_name):
            print("read_csv:%s" % file_name)
            load_df = pd.read_csv(file_name)
        else:
            load_df = pro.stock_basic(exchange = '', list_status = 'L', fields = 'ts_code,symbol,name,area,industry,list_date')
            load_df.to_csv(file_name)

        load_df = load_df[load_df['list_date'] <= int(input_stocks_list_end_date)]
        load_df = load_df.copy()
        load_df = load_df.reset_index(drop=True)

        industry_filter_en = False
        code_filter_en = False
        circ_mv_filter_en = False
        if input_industry_filter != '':
            industry_list = input_industry_filter.split(',')
            industry_filter_en = True
        if input_code_filter != '':
            code_filter_list = input_code_filter.split(',')
            code_filter_en = True
        if min_circ_mv > 0 or max_circ_mv > 0:
            circ_mv_filter_en = True
            DownloadATradeDayDataDailyBasic('20190510')
            daily_basic_df = LoadATradeDayDataDailyBasic('20190510')

        code_valid_list = []
        for iloop in range(0, len(load_df)):
            temp_code = load_df['ts_code'][iloop]
            temp_code_valid = True
            if industry_filter_en:
                if not load_df['industry'][iloop] in industry_list:
                    temp_code_valid = False
            if code_filter_en:
                if not StockCodeFilter(temp_code, code_filter_list):
                    temp_code_valid = False
            if circ_mv_filter_en:
                find_df = daily_basic_df[daily_basic_df['ts_code'] == temp_code]
                find_df = find_df.copy()
                find_df = find_df.reset_index(drop=True)
                if len(find_df) == 0:
                    temp_code_valid = False
                else:
                    temp_circ_mv = find_df.loc[0, 'circ_mv']
                    if min_circ_mv > 0 and temp_circ_mv < min_circ_mv:
                        temp_code_valid = False
                    if max_circ_mv > 0 and temp_circ_mv > max_circ_mv:
                        temp_code_valid = False
                
            code_valid_list.append(temp_code_valid)
        load_df = load_df[code_valid_list]
        print(load_df)
        print('StockCodes(%s)[%u]' % (input_industry_filter, len(load_df)))
        code_list = load_df['ts_code'].values
        name_list = load_df['name'].values
        return code_list, name_list

def StockCodes():
    code_list, name_list = StockCodesName(stocks_list_end_date, industry_filter, code_filter)
    return code_list

def StockName(ts_code):
    code_list, name_list = StockCodesName(stocks_list_end_date, industry_filter, code_filter)
    for iloop in range(0, len(code_list)):
        if code_list[iloop] == ts_code:
            return name_list[iloop]
    return 'unknow'

def FileNameStockDownloadDataDaily(stock_code):
    temp_file_name = './data/daily/' + stock_code + '_' + train_test_date + '.csv'
    return temp_file_name

def FileNameStockDownloadDataDailyBasic(stock_code):
    temp_file_name = './data/daily_basic/' + stock_code + '_' + train_test_date + '.csv'
    return temp_file_name

def FileNameStockDownloadDataMoneyFlow(stock_code):
    temp_file_name = './data/moneyflow/' + stock_code + '_' + train_test_date + '.csv'
    return temp_file_name

def FileNameStockDownloadDataAdjFactor(stock_code):
    temp_file_name = './data/adj_factor/' + stock_code + '_' + train_test_date + '.csv'
    return temp_file_name

def FileNameTradeDayDownloadDataDaily(trade_date):
    temp_file_name = './data/daily/'+'trade_date'+'_'+trade_date+'.csv'
    return temp_file_name

def FileNameTradeDayDownloadDataDailyBasic(trade_date):
    temp_file_name = './data/daily_basic/'+'trade_date'+'_'+trade_date+'.csv'
    return temp_file_name

def FileNameTradeDayDownloadDataMoneyFlow(trade_date):
    temp_file_name = './data/moneyflow/'+'trade_date'+'_'+trade_date+'.csv'
    return temp_file_name

def FileNameTradeDayDownloadDataAdjFactor(trade_date):
    temp_file_name = './data/adj_factor/'+'trade_date'+'_'+trade_date+'.csv'
    return temp_file_name

# 对于PP data，只关注stock code、结束时间
def FileNameStockPreprocessedData(stock_code):
    temp_file_name = './data/preprocessed/%s_%s_%s_%u_%u.csv' %(\
        stock_code, \
        pp_data_start_date, \
        train_test_date, \
        int(use_daily_basic), \
        int(use_money_flow))
    return temp_file_name

def FileNameMergePPDataOriginal():
    file_name = './data/preprocessed/merge_%s_%s_%s_%s_%s.npy' % ( \
        stocks_list_end_date, \
        pp_data_start_date, \
        train_test_date, \
        industry_filter, \
        code_filter)
    return file_name

def FileNameMergePPDataOriginalSimplify():
    file_name = './data/preprocessed/merge_simplify_%s_%s_%s_%s_%s.npy' % ( \
        stocks_list_end_date, \
        pp_data_start_date, \
        train_test_date, \
        industry_filter, \
        code_filter)
    return file_name

def FileNameMergePPData(input_date):
    file_name = './data/preprocessed/merge_%s_%s_%s_%s_%s.npy' % ( \
        stocks_list_end_date, \
        pp_data_start_date, \
        input_date, \
        industry_filter, \
        # '')
        code_filter)
    return file_name

def DownloadAStocksDataDaily(ts_code):
    pro = ts.pro_api()
    start_date = '19000101'
    end_data = train_test_date
    file_name = FileNameStockDownloadDataDaily(ts_code)
    if not os.path.exists(file_name):
        df_daily = pro.daily(ts_code = ts_code, start_date = start_date, end_date = end_data)
        df_daily.to_csv(file_name)

def DownloadAStocksDataDailyBasic(ts_code):
    pro = ts.pro_api()
    start_date = '19000101'
    end_data = train_test_date
    file_name = FileNameStockDownloadDataDailyBasic(ts_code)
    if not os.path.exists(file_name):
        if DATA_TYPE_DAY == data_type:
            # df_basic=pro.daily_basic(ts_code = ts_code, start_date = start_date, end_date = end_data)
            # df = pro.daily(ts_code = ts_code, start_date = start_date, end_date = end_data)
            # if len(df_basic) != len(df) :
            #     print("DownloadAStocksData.error.1")
            #     return
            # df.drop(['close','ts_code'],axis=1,inplace=True)
            # df_merge = pd.merge(df_basic, df, left_on='trade_date', right_on='trade_date')
            # df_merge.to_csv(file_name)

            df_basic = pro.daily_basic(ts_code = ts_code, start_date = start_date, end_date = end_data)
            df_basic.to_csv(file_name)
        elif DATA_TYPE_WEEK == data_type:
            df_week = pro.weekly(ts_code = ts_code, start_date = start_date, end_date = end_data)
            df_week.to_csv(file_name)

def DownloadAStocksDataMoneyFlow(ts_code):
    pro = ts.pro_api()
    start_date = '19000101'
    end_data = train_test_date
    file_name = FileNameStockDownloadDataMoneyFlow(ts_code)
    if not os.path.exists(file_name):
        df_moneyflow = pro.moneyflow(ts_code = ts_code, start_date = start_date, end_date = end_data)
        df_moneyflow.to_csv(file_name)

def DownloadAStocksDataAdjFactor(ts_code):
    pro = ts.pro_api()
    start_date = '19000101'
    end_data = train_test_date
    file_name = FileNameStockDownloadDataAdjFactor(ts_code)
    if not os.path.exists(file_name):
        df = pro.adj_factor(ts_code = ts_code, trade_date='')
        col_date = df['trade_date'].copy()
        col_date = pd.to_numeric(col_date)
        df = df[col_date <= int(end_data)].copy().reset_index(drop=True)
        df.to_csv(file_name)

def DownloadAStocksData(ts_code):
    DownloadAStocksDataDaily(ts_code)
    if use_daily_basic:
        DownloadAStocksDataDailyBasic(ts_code)
    if use_money_flow:
        DownloadAStocksDataMoneyFlow(ts_code)
    if use_adj_factor:
        DownloadAStocksDataAdjFactor(ts_code)

def DownloadATradeDayDataDailyBasic(input_trade_date):
    pro = ts.pro_api()
    file_name = FileNameTradeDayDownloadDataDailyBasic(input_trade_date)
    if not os.path.exists(file_name):
        df_basic = pro.daily_basic(trade_date=input_trade_date)
        df_basic.to_csv(file_name)

def DownloadATradeDayDataDaily(input_trade_date):
    pro = ts.pro_api()
    file_name = FileNameTradeDayDownloadDataDaily(input_trade_date)
    while True:
        if os.path.exists(file_name):
            temp_df = pd.read_csv(file_name)
            if len(temp_df) == 0:
                os.remove(file_name)
            else:
                break;
        if not os.path.exists(file_name):
            print('download daily %s' % input_trade_date)
            df_basic = pro.daily(trade_date=input_trade_date)
            df_basic.to_csv(file_name)

def DownloadATradeDayDataAdjFactor(input_trade_date):
    pro = ts.pro_api()
    file_name = FileNameTradeDayDownloadDataAdjFactor(input_trade_date)
    while True:
        if os.path.exists(file_name):
            temp_df = pd.read_csv(file_name)
            if len(temp_df) == 0:
                os.remove(file_name)
            else:
                break;
        if not os.path.exists(file_name):
            print('download adj_factor %s' % input_trade_date)
            df_basic = pro.adj_factor(ts_code='', trade_date=input_trade_date)
            df_basic.to_csv(file_name)

def DownloadATradeDayDataMoneyFlow(input_trade_date):
    pro = ts.pro_api()
    file_name = FileNameTradeDayDownloadDataMoneyFlow(input_trade_date)
    if not os.path.exists(file_name):
        df_basic = pro.moneyflow(trade_date=input_trade_date)
        df_basic.to_csv(file_name)
    
def DownloadATradeDayData(input_trade_date):
    DownloadATradeDayDataDaily(input_trade_date)
    if use_adj_factor:
        DownloadATradeDayDataAdjFactor(input_trade_date)
    # DownloadATradeDayDataDailyBasic(input_trade_date)
    # DownloadATradeDayDataMoneyFlow(input_trade_date)

def LoadATradeDayData(trade_date):
    file_name = FileNameTradeDayDownloadDataDaily(trade_date)
    if not os.path.exists(file_name):
        return pd.DataFrame()
    merge_df = pd.read_csv(file_name)
    if use_adj_factor:
        file_name = FileNameTradeDayDownloadDataAdjFactor(trade_date)
        if not os.path.exists(file_name):
            return pd.DataFrame()
        adj_df = pd.read_csv(file_name)
        merge_df = TradeDayDataMerge(merge_df, adj_df)
    return merge_df
    # file_name = FileNameTradeDayDownloadDataMoneyFlow(trade_date)
    # moneyflow_df = pd.read_csv(file_name)
    # return StockDataMerge(daily_df, moneyflow_df)

def LoadATradeDayDataDailyBasic(trade_date):
    file_name = FileNameTradeDayDownloadDataDailyBasic(trade_date)
    daily_df = pd.read_csv(file_name)
    return daily_df

def DownloadTrainTestData():
    code_list = StockCodes()
    for code_index in range(0, len(code_list)):
        stock_code = code_list[code_index]
        DownloadAStocksData(stock_code)
        print("%-4d : %s 100%%" % (code_index, stock_code))

def PredictTradeDateList():
    ref_trade_day_num = feature_relate_days + referfence_feature_count + 30 - 1 + 20  # 允许有20天停盘
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

def StockDataMerge(df_1, df_2):
    df_2.drop(['ts_code'],axis=1,inplace=True)
    df_merge = pd.merge(df_1, df_2, left_on='trade_date', right_on='trade_date')
    return df_merge

def TradeDayDataMerge(df_1, df_2):
    df_2.drop(['trade_date'],axis=1,inplace=True)
    df_merge = pd.merge(df_1, df_2, left_on='ts_code', right_on='ts_code')
    return df_merge

# 返回 result, avg_value
def GetAvg(src_df, day_pointer, target_name, avg_period):
    if (len(src_df) - day_pointer) < avg_period:
        return False, 0.0
    sum_value = 0.0
    for iloop in range(0, avg_period):
        day_index = day_pointer + iloop
        temp_value = src_df.loc[day_index, target_name]
        if math.isnan(temp_value) or temp_value <= 0.0:
            ts_code = src_df.loc[0, 'ts_code']
            print('Error, GetAvg(%s, %u, %u, %s), isnan' % (ts_code, day_pointer, iloop, target_name))
            return False, 0.0
        sum_value += temp_value
    avg_value = sum_value / avg_period
    return True, avg_value



def GetTrainDataCaption():
    caption = []
    if feature_type == FEATURE_G7_10D8:
        caption.append('total_share')
        caption.append('float_share')
        caption.append('free_share')
        caption.append('total_mv')
        caption.append('circ_mv')
        caption.append('close')
        caption.append('close_5_avg')
        for iloop in range(0, 10):                
            caption.append('open_increase_pre_%u' % iloop)
            caption.append('close_increase_pre_%u' % iloop)
            caption.append('high_increase_pre_%u' % iloop)
            caption.append('low_increase_pre_%u' % iloop)
            caption.append('close_to_close_5_avg_pre_%u' % iloop)
            caption.append('close_to_close_10_avg_pre_%u' % iloop)
            caption.append('close_to_close_30_avg_pre_%u' % iloop)
            caption.append('turnover_rate_f_pre_%u' % iloop)
    elif feature_type == FEATURE_G2_10D8:
        caption.append('close')
        caption.append('close_5_avg')
        for iloop in range(0, 10):                
            caption.append('open_increase_pre_%u' % iloop)
            caption.append('close_increase_pre_%u' % iloop)
            caption.append('high_increase_pre_%u' % iloop)
            caption.append('low_increase_pre_%u' % iloop)
            caption.append('close_to_close_5_avg_pre_%u' % iloop)
            caption.append('close_to_close_10_avg_pre_%u' % iloop)
            caption.append('close_to_close_30_avg_pre_%u' % iloop)
            caption.append('turnover_rate_f_pre_%u' % iloop)
    elif feature_type == FEATURE_G2_10D2:
        caption.append('total_share')
        caption.append('float_share')
        for iloop in range(0, 10):                
            caption.append('close_increase_pre_%u' % iloop)
            caption.append('turnover_rate_f_pre_%u' % iloop)
    elif feature_type == FEATURE_G0_10D2:
        for iloop in range(0, 10):                
            caption.append('close_pre_%u' % iloop)
            caption.append('open_pre_%u' % iloop)
    elif feature_type == FEATURE_G7_10AVG102_10D8:
        temp_index = feature_day_pointer
        caption.append('total_share')
        caption.append('float_share')
        caption.append('free_share')
        caption.append('total_mv')
        caption.append('circ_mv')
        caption.append('close')
        caption.append('close_5_avg')
        for iloop in range(0, 10):                
            caption.append('close_10_avg_pre_%u' % iloop * 10)
            caption.append('turnover_rate_f_10_avg_pre_%u' % iloop * 10)
        for iloop in range(0, 10):                
            caption.append('open_increase_pre_%u' % iloop)
            caption.append('close_increase_pre_%u' % iloop)
            caption.append('high_increase_pre_%u' % iloop)
            caption.append('low_increase_pre_%u' % iloop)
            caption.append('close_to_close_5_avg_pre_%u' % iloop)
            caption.append('close_to_close_10_avg_pre_%u' % iloop)
            caption.append('close_to_close_30_avg_pre_%u' % iloop)
            caption.append('turnover_rate_f_pre_%u' % iloop)
    elif feature_type == FEATURE_G0_10D5:
        for iloop in range(0, 10):            
            caption.append('open_pre_%u' % iloop)
            caption.append('close_pre_%u' % iloop)
            caption.append('high_pre_%u' % iloop)
            caption.append('low_pre_%u' % iloop)
            caption.append('turnover_rate_f_pre_%u' % iloop)    
    elif feature_type == FEATURE_G0_10D5_TO_100_AVG:
        for iloop in range(0, 10):           
            caption.append('open_to_close_100_avg_pre_%u' % iloop)
            caption.append('close_to_close_100_avg_pre_%u' % iloop)
            caption.append('high_to_close_100_avg_pre_%u' % iloop)
            caption.append('low_to_close_100_avg_pre_%u' % iloop)
            caption.append('vol_to_vol_100_avg_pre_%u' % iloop)     
    elif feature_type == FEATURE_G0_10W5_TO_100_AVG:
        for iloop in range(0, 10):        
            caption.append('open_5_to_close_100_avg_pre_%u' % iloop)
            caption.append('close_5_to_close_100_avg_pre_%u' % iloop)
            caption.append('high_5_to_close_100_avg_pre_%u' % iloop)
            caption.append('low_5_to_close_100_avg_pre_%u' % iloop)
            caption.append('vol_5_to_vol_100_avg_pre_%u' % iloop)        
    elif feature_type == FEATURE_G0_100D5_TO_100_AVG:
        for iloop in range(0, 100):                
            caption.append('open_to_close_100_avg_pre_%u' % iloop)
            caption.append('close_to_close_100_avg_pre_%u' % iloop)
            caption.append('high_to_close_100_avg_pre_%u' % iloop)
            caption.append('low_to_close_100_avg_pre_%u' % iloop)
            caption.append('vol_to_vol_100_avg_pre_%u' % iloop)   
    elif feature_type == FEATURE_G0_100D2_TO_100_AVG:
        for iloop in range(0, 100):                
            caption.append('close_to_close_100_avg_pre_%u' % iloop)
            caption.append('vol_to_vol_100_avg_pre_%u' % iloop)    

    for iloop in range(0, max_predict_day_count):
        caption.append('label_%u' % iloop)

    caption.append('act_open_increase')
    caption.append('act_low_increase')
    caption.append('act_open')
    caption.append('act_low')
    caption.append('act_close')
    caption.append('act_tscode')
    caption.append('act_date')
    return caption
    

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


# def GetDataCaption(feature_type):
#     data_unit.append(src_df['total_share'][temp_index])
#         data_unit.append(src_df['float_share'][temp_index])
#         data_unit.append(src_df['free_share'][temp_index])
#         data_unit.append(src_df['total_mv'][temp_index])
#         data_unit.append(src_df['circ_mv'][temp_index])
#         data_unit.append(src_df['close'][temp_index])
#         data_unit.append(src_df['close_5_avg'][temp_index])
#         for iloop in range(0, 10):                
#             temp_index=feature_day_pointer+iloop
#             data_unit.append(src_df['open_increase'][temp_index])
#             data_unit.append(src_df['close_increase'][temp_index])
#             data_unit.append(src_df['high_increase'][temp_index])
#             data_unit.append(src_df['low_increase'][temp_index])
#             data_unit.append(src_df['close_increase_to_5_avg'][temp_index])
#             data_unit.append(src_df['close_increase_to_10_avg'][temp_index])
#             data_unit.append(src_df['close_increase_to_30_avg'][temp_index])
#             data_unit.append(src_df['turnover_rate_f'][temp_index])
#     if feature_type == FEATURE_TYPE_TRAIN:
#         return [\
#         'total_share', 'float_share', 'free_share', 'total_mv', 'circ_mv', 'close', 'close_5_avg', \
#         'open_increase_0', 'close_increase_0', 'high_increase_0', 'low_increase_0', 
#         ]


# 对于整体训练数据，关注
# feature size
# label type
# max_predict_day_count
# 上市日期门限、
# 训练数据起始日期、
# 训练数据截至日期、
# 个股训练数据最小和最大数据量
def FileNameTrainData():
    file_name = './data/dataset/train_data_%u_%u_%u_%s_%s_%s_%u_%u_%s%s.npy' % ( \
        feature_type, \
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

def SettingName():
    temp_name = '%u_%u_%u_%u_%s_%s_%s_%u_%u_%s%s' % ( \
        feature_type, \
        label_type, \
        max_predict_day_count, \
        predict_day_count, \
        stocks_list_end_date, \
        train_data_start_date, \
        train_data_end_date, \
        train_a_stock_min_data_num, \
        train_a_stock_max_data_num, \
        industry_filter, \
        code_filter)
    return temp_name

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
    file_name = './data/dataset/test_data_%u_%u_%u_%u_%s_%s_%s_%u_%d_%s%s.npy' % ( \
        feature_type, \
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

def OffsetTradeDate(ref_date, day_offset):
    temp_list = TradeDateList(ref_date, day_offset + 1)
    return temp_list[day_offset]

def UpdatePreprocessDataAStock(code_index, stock_code):
    stock_pp_file_name = FileNameStockPreprocessedData(stock_code)
    if not os.path.exists(stock_pp_file_name):
        file_name_daily = FileNameStockDownloadDataDaily(stock_code)
        if not os.path.exists(file_name_daily):
            return
        merge_df = pd.read_csv(file_name_daily)
        if use_daily_basic:
            file_name_daily_basic = FileNameStockDownloadDataDailyBasic(stock_code)
            if not os.path.exists(file_name_daily_basic):
                return
            df_daily_basic = pd.read_csv(file_name_daily_basic)
            df_daily_basic.drop(['close'],axis=1,inplace=True)
            merge_df = StockDataMerge(merge_df, df_daily_basic)
        if use_money_flow:
            file_name_money_flow = FileNameStockDownloadDataMoneyFlow(stock_code)
            if not os.path.exists(file_name_money_flow):
                return
            df_money_flow = pd.read_csv(file_name_money_flow)
            merge_df = StockDataMerge(merge_df, df_money_flow)
        if use_adj_factor:
            file_name_adj_factor = FileNameStockDownloadDataAdjFactor(stock_code)
            if not os.path.exists(file_name_adj_factor):
                return
            df_adj_factor = pd.read_csv(file_name_adj_factor)
            merge_df = StockDataMerge(merge_df, df_adj_factor)
        
        merge_df = merge_df[merge_df['trade_date'] >= int(pp_data_start_date)]
        pp_data = preprocess.StockDataPreProcess(merge_df)
        if len(pp_data) > 0:
            pp_data.to_csv(stock_pp_file_name)
            if code_index > 0:
                print("%-4d : %s 100%%" % (code_index, stock_code))
        else:
            if code_index > 0:
                print("%-4d : %s error" % (code_index, stock_code))
    else:
        if code_index > 0:
            print("%-4d : %s 100%%" % (code_index, stock_code))

def UpdatePreprocessData():
    code_list = StockCodes()
    for code_index in range(0, len(code_list)):
        stock_code = code_list[code_index]
        UpdatePreprocessDataAStock(code_index, stock_code)

def CheckPreprocessDataAStock(stock_pp_file_name):
    if os.path.exists(stock_pp_file_name):
        src_df = pd.read_csv(stock_pp_file_name)
        for day_loop in range(0, len(src_df)):
            if src_df.loc[day_loop,'trade_date'] == '' \
                    or src_df.loc[day_loop,'open'] == 0.0 \
                    or src_df.loc[day_loop,'close'] == 0.0 \
                    or src_df.loc[day_loop,'high'] == 0.0 \
                    or src_df.loc[day_loop,'low'] == 0.0:
                print('CheckPreprocessDataAStock.Error1, %s, %s, %f, %f, %f, %f' %( \
                                                                        src_df.loc[day_loop,'ts_code'], \
                                                                        src_df.loc[day_loop,'trade_date'], \
                                                                        src_df.loc[day_loop,'open'], \
                                                                        src_df.loc[day_loop,'close'], \
                                                                        src_df.loc[day_loop,'high'], \
                                                                        src_df.loc[day_loop,'low']))
                return False
            if math.isnan(src_df.loc[day_loop,'trade_date']) \
                    or math.isnan(src_df.loc[day_loop,'open']) \
                    or math.isnan(src_df.loc[day_loop,'close']) \
                    or math.isnan(src_df.loc[day_loop,'high']) \
                    or math.isnan(src_df.loc[day_loop,'low']) \
                    or math.isnan(src_df.loc[day_loop,'turnover_rate_f']) \
                    or math.isnan(src_df.loc[day_loop,'turnover_rate_f_10_avg']):
                print('CheckPreprocessDataAStock.Error2, %s, %s, %f, %f, %f, %f, %f, %f' %( \
                                                                        src_df.loc[day_loop,'ts_code'], \
                                                                        src_df.loc[day_loop,'trade_date'], \
                                                                        src_df.loc[day_loop,'open'], \
                                                                        src_df.loc[day_loop,'close'], \
                                                                        src_df.loc[day_loop,'high'], \
                                                                        src_df.loc[day_loop,'low'], \
                                                                        src_df.loc[day_loop,'turnover_rate_f'], \
                                                                        src_df.loc[day_loop,'turnover_rate_f_10_avg']))
                return False
        return True
    else:
        print('CheckPreprocessDataAStock.Error3, %s' % stock_pp_file_name)
        return False

def CheckPreprocessData():
    code_list = StockCodes()
    for code_index in range(0, len(code_list)):
        stock_code = code_list[code_index]
        stock_pp_file_name = FileNameStockPreprocessedData(stock_code)
        if os.path.exists(stock_pp_file_name):
            if CheckPreprocessDataAStock(stock_pp_file_name):
                print("%-4d : %s 100%%" % (code_index, stock_code))

def GetTrainData():
    train_data = np.load(FileNameTrainData())
    print("train_data: {}".format(train_data.shape))
    # raw_input("Enter ...")

    print("reorder...")
    order=np.argsort(np.random.random(len(train_data)))
    train_data=train_data[order]
    train_data=train_data[:2000000]
    # raw_input("Enter ...")
    sample_train_data = train_data[:10]

    label_index = label_col_index
    print("get feature ...")
    train_features = train_data[:, 0:feature_size].copy()
    # raw_input("Enter ...")

    print("get label...")
    train_labels = train_data[:, label_index:label_index+1].copy()
    # raw_input("Enter ...")
    print("train_features: {}".format(train_features.shape))
    print("train_labels: {}".format(train_labels.shape))

    # caption = GetTrainDataCaption()
    # print('caption[%u]:' % len(caption))
    # print(caption)
    
    # sample_train_data_df = pd.DataFrame(sample_train_data, columns=caption)
    # sample_train_data_df.to_csv('./sample_train_data_df.csv')
    return train_features, train_labels

def GetTrainDataBalance(label_threshold, neg_ratio):
    train_data = np.load(FileNameTrainData())
    print("train_data: {}".format(train_data.shape))
    # raw_input("Enter ...")

    label_index = label_col_index
    sort_order = train_data[:,label_index].argsort()
    sort_order = sort_order[::-1]
    train_data = train_data[sort_order]
    pos_mask = train_data[:,label_index] >= label_threshold
    pos_mask = pos_mask[pos_mask]
    pos_num = len(pos_mask)
    train_data = train_data[:(1+neg_ratio)*pos_num]

    print("reorder...")
    order=np.argsort(np.random.random(len(train_data)))
    train_data=train_data[order]
    train_data=train_data[:2000000]
    # raw_input("Enter ...")

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
        if(len(processed_df) >= (feature_relate_days)):
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

# def GetAStockFeatures(ts_code, input_date):
#     pro = ts.pro_api()
#     temp_file_name = './download_data/' + ts_code + '_' + input_date + '.csv'
#     if os.path.exists(temp_file_name):
#         df_merge = pd.read_csv(temp_file_name)
#     else:
#         start_date = '19000101'
#         df_basic=pro.daily_basic(ts_code = ts_code, start_date = start_date, end_date = input_date)
#         df = pro.daily(ts_code = ts_code, start_date = start_date, end_date = input_date)
#         if len(df_basic) != len(df) :
#             print("DownloadAStocksData.error.1")
#             return
#         df.drop(['close','ts_code'],axis=1,inplace=True)
#         df_merge = pd.merge(df_basic, df, left_on='trade_date', right_on='trade_date')
#         df_merge.to_csv(temp_file_name)

#     pp_data = StockDataPreProcess(df_merge)
#     test_data_list = []
#     for day_loop in range(0, 20):
#         data_unit = GetAFeature(pp_data, day_loop, FEATURE_TYPE_PREDICT)
#         test_data_list.append(data_unit)
#     np_data = np.array(test_data_list)
#     return np_data






fut_list_end_date = '20190101'
fut_industry_filter = '豆粕'
def FutIndustryFilter(ts_name, industry_filter_list):
    for it in industry_filter_list:
        if ts_name[0:len(it)] == it:
            return True
    return False

def FutCodes():
    pro = ts.pro_api()

    file_name = './download_data_fut/' + 'ts_code' + '.csv'
    # if not os.path.exists(file_name):
    load_df = pro.fut_basic(exchange='DCE', fields='ts_code,symbol,name,list_date,delist_date')
    load_df.to_csv(file_name)
    print("read_csv:%s" % file_name)
    load_df = pd.read_csv(file_name)

    load_df = load_df[load_df['list_date'] <= int(fut_list_end_date)]
    load_df = load_df.copy()
    load_df = load_df.reset_index(drop=True)

    industry_filter_en = False
    if fut_industry_filter != '':
        industry_list = fut_industry_filter.split(',')
        industry_filter_en = True

    code_valid_list = []
    for iloop in range(0, len(load_df)):
        temp_code_valid = True
        if industry_filter_en:
            if not FutIndustryFilter(load_df['name'][iloop], industry_list):
                temp_code_valid = False
        code_valid_list.append(temp_code_valid)
    load_df = load_df[code_valid_list]
    print(load_df)
    print('StockCodes(%s)[%u]' % (industry_filter, len(load_df)))
    code_list = load_df['ts_code'].values
    return code_list

def FileNameFutDownloadData(ts_code):
    temp_file_name = './download_data_fut/' + ts_code + '_' + train_test_date + '.csv'
    return temp_file_name

def DownloadAFutData(ts_code):
    pro = ts.pro_api()
    start_date = '19000101'
    end_data = train_test_date
    file_name = FileNameFutDownloadData(ts_code)
    if not os.path.exists(file_name):
        df = pro.fut_daily(ts_code = ts_code, start_date = start_date, end_date = end_data)
        df.to_csv(file_name)

def DownloadFutTrainTestData():
    code_list = FutCodes()
    for code_index in range(0, len(code_list)):
        stock_code = code_list[code_index]
        DownloadAFutData(stock_code)
        print("%-4d : %s 100%%" % (code_index, stock_code))

def UpdateFutTrainTestData():
    if os.path.exists(FileNameTrainData()) and os.path.exists(FileNameTestData()):
        return
    code_list = FutCodes()
    train_data_init_flag = True
    test_data_init_flag = True
    train_pp_start_date = OffsetTradeDate(train_data_start_date, feature_relate_days + max_predict_day_count)
    train_pp_end_date = train_data_end_date
    test_pp_start_date = OffsetTradeDate(test_data_start_date, feature_relate_days + max_predict_day_count)
    test_pp_end_date = test_data_end_date
    for code_index in range(0, len(code_list)):
        fut_code = code_list[code_index]
        fut_download_file_name = FileNameFutDownloadData(stock_code)
        if os.path.exists(fut_download_file_name):
            pp_data = pd.read_csv(fut_download_file_name)
        
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
            valid_data_num = len(train_pp_data) - feature_relate_days - max_predict_day_count
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
            valid_data_num = len(test_pp_data) - feature_relate_days - max_predict_day_count
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

def Features10D14To10D5(features):
    for iloop in range(0, 10):
        temp_offset = iloop * 14
        temp_featrue = features[:,  temp_offset: temp_offset + 5]
        if iloop == 0:
            output_features = temp_featrue
        else:
            output_features = np.append(output_features, temp_featrue, axis=1)
    return output_features

def Test():
    # pro = ts.pro_api()
    # df = pro.fut_basic(exchange='SHFE', fut_type='1', fields='ts_code,symbol,name,list_date,delist_date')
    # print(df)
    # df.to_csv('./download_data_fut/ts_code.csv')

    # temp_code = 'V1109.DCE'
    # df = pro.fut_daily(ts_code = temp_code)
    # df.to_csv('./download_data_fut/%s.csv' % temp_code)

    # temp_code = 'V1110.DCE'
    # df = pro.fut_daily(ts_code = temp_code)
    # df.to_csv('./download_data_dce/%s.csv' % temp_code)
    # fut_codes = FutCodes()
    pp_data = pd.read_csv('./hk_stock_data/00001.csv', encoding = 'gbk')
    print(pp_data)

if __name__ == "__main__":
    # temp_stock_codes=StockCodes()
    # print("temp_stock_codes:")
    # for iloop in range(0,len(temp_stock_codes)):
    #     print("%-4d : %s" % (iloop, temp_stock_codes[iloop]))
    # print("\n\n\n")
    # Test()

    # stock_code = '000119.HK'
    # stock_download_file_name = FileNameStockDownloadData(stock_code)
    # stock_pp_file_name = FileNameStockPreprocessedData(stock_code)
    # download_df = pd.read_csv(stock_download_file_name)
    # pp_data = StockDataPreProcess(download_df)
    # print('len(pp_data) : %d' % len(pp_data))
    # if len(pp_data) > 0:
    #     pp_data.to_csv(stock_pp_file_name)

    # DownloadAStocksDataMoneyFlow('600050.SH')

    # train_test_date = CurrentDate()
    # DownloadAStocksData('600050.SH')
    # UpdatePreprocessDataAStock(0, '600050.SH')

    # df = pro.adj_factor(ts_code='600050.SH', trade_date='')
    # df.to_csv('./temp.csv')
    # print(df)

    DownloadAStocksDataAdjFactor('600050.SH')

