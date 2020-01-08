# -*- coding:UTF-8 -*-


import tushare as ts
import numpy as np
import pandas as pd
import os
import shutil
import time
import datetime
import sys
import math
import preprocess
sys.path.append("..")
from common import base_common
from common import np_common

pro = ts.pro_api()

# preprocess_ref_days = 100

reload(sys)
sys.setdefaultencoding('utf-8')
# max_predict_day_count = 10  # 决定train_data 和 test_data 的predict_day_count
# predict_day_count = 2  # 预测未来几日的数据
# referfence_feature_count = 1
# test_acture_data_with_feature = False
# train_a_stock_min_data_num = 400
# train_a_stock_max_data_num = 1000000

use_daily_basic = False
use_money_flow = False
use_adj_factor = True



# pp_data_start_date = '20000101'
# stocks_list_end_date = '20000101'
# train_data_start_date = '20120101'
# train_data_end_date = '20170101'
# test_data_start_date = '20170101'
# test_data_end_date = '20190414'
# train_test_date = '20190414'
# predict_date = '20181225'



ts.set_token('230c446ae448ec95357d0f7e804ddeebc7a51ff340b4e6e0913ea2fa')

pd.set_option('display.width', 150)  # 设置字符显示宽度
pd.set_option('display.max_rows', 20)  # 设置显示最大行

def TradeDateLowLevel(end_date): 
    file_name = './data/trade_date/trade_date_%s.npy' % str(end_date)
    if os.path.exists(file_name):
        date_np = np.load(file_name)
        return date_np
    else:
        df_trade_cal = pro.trade_cal(exchange = 'SSE', start_date = '19800101', end_date = str(end_date))
        df_trade_cal = df_trade_cal.sort_index(ascending = False)
        df_trade_cal = df_trade_cal[df_trade_cal['is_open'] == 1]
        date_np = df_trade_cal['cal_date'].values
        base_common.MKFileDirs(file_name)
        np.save(file_name, date_np)
    return date_np

def TradeDateList(end_date, trade_day_num):
    date_np = TradeDateLowLevel(end_date)
    return date_np[:trade_day_num].copy().tolist()

def TradeDateListRange(start_date, end_date):
    date_np = TradeDateLowLevel(end_date)
    pos = date_np.astype(np.int64) >= int(start_date)
    return date_np[pos].copy().tolist()

def StockCodeFilter(ts_code, code_filter_list):
    for it in code_filter_list:
        if ts_code[0:len(it)] == it:
            return True
    return False

def StockCodesName(release_end_date, industry_filter, code_filter, sample_step):
    file_name = './data/' + 'stock_code' + '.csv'
    if os.path.exists(file_name):
        load_df = pd.read_csv(file_name)
    else:
        load_df = pro.stock_basic(exchange = '', list_status = 'L', fields = 'ts_code,symbol,name,area,industry,list_date')
        base_common.MKFileDirs(file_name)
        load_df.to_csv(file_name)

    if release_end_date > 0:
        load_df = load_df[load_df['list_date'] <= int(release_end_date)]
    if sample_step > 1:
        load_df = load_df[::sample_step]
    load_df = load_df.copy()
    load_df = load_df.reset_index(drop=True)

    industry_filter_en = False
    code_filter_en = False
    circ_mv_filter_en = False
    if industry_filter != '':
        industry_list = industry_filter.split(',')
        industry_filter_en = True
    if code_filter != '':
        code_filter_list = code_filter.split(',')
        code_filter_en = True

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
            
        code_valid_list.append(temp_code_valid)
    load_df = load_df[code_valid_list]
    # print(load_df)
    code_list = load_df['ts_code'].values
    name_list = load_df['name'].values
    return code_list, name_list

def StockCodes(release_end_date, industry_filter, code_filter, sample_step):
    code_list, name_list = StockCodesName(release_end_date, industry_filter, code_filter, sample_step)
    return code_list

def StockName(ts_code):
    code_list, name_list = StockCodesName(release_end_date, industry_filter, code_filter, sample_step)
    for iloop in range(0, len(code_list)):
        if code_list[iloop] == ts_code:
            return name_list[iloop]
    return 'unknow'


def UpdatePPDataMTFunc(param, msg):
    param.UpdateStockPPData(msg)

class DataSource():
    def __init__(self, 
                 release_end_date, 
                 industry_filter, 
                 code_filter, 
                 sample_step, 
                 start_date, 
                 end_date, 
                 use_daily_basic, 
                 use_money_flow, 
                 use_adj_factor,
                 adj_mode='f'):
        self.release_end_date = release_end_date
        self.industry_filter = industry_filter
        self.code_filter = code_filter
        self.sample_step = sample_step
        self.start_date = start_date
        self.end_date = end_date
        self.use_daily_basic = use_daily_basic
        self.use_money_flow = use_money_flow
        self.use_adj_factor = use_adj_factor
        self.adj_mode = adj_mode
        self.download_data_name_list = ['daily', 'daily_basic', 'moneyflow', 'adj_factor']
        self.setting_name_stock_pp = '%s_%u_%u_%u_%s' % (\
                            str(end_date), \
                            int(use_daily_basic), \
                            int(use_money_flow), \
                            int(use_adj_factor), \
                            adj_mode)
        self.setting_name_stock = '%s_%s' % (\
                            str(start_date), \
                            self.setting_name_stock_pp)
        self.setting_name = '%s_%s_%s_%u_%s' % (\
                            str(release_end_date), \
                            industry_filter, \
                            code_filter, \
                            sample_step, \
                            self.setting_name_stock)
        self.code_list = StockCodes(self.release_end_date, self.industry_filter, self.code_filter, self.sample_step)
        self.date_list = TradeDateListRange(self.start_date, self.end_date)
        self.code_index_map = base_common.ListToIndexMap(self.code_list, False)

    def ShowStockCodes(self):
        print('%-7s%s' % ('index', 'ts_code'))
        print('-' * 32)
        for iloop in range(len(self.code_list)):
            print('%-7u%s' % (iloop, self.code_list[iloop]))
        print('\n')

    def FileNameStockDownloadData(self, ts_code):
        name_list = []
        for data_name in self.download_data_name_list:
            name_list.append('./data/%s/%s_%s.csv' % (data_name, ts_code, str(self.end_date)))
        return name_list

    def FileNameTradeDayDownloadData(self, trade_date):
        name_list = []
        for data_name in self.download_data_name_list:
            name_list.append('./data/%s/trade_date_%s.csv' % (data_name, str(trade_date)))
        return name_list

    def FileNameStockPPData(self, ts_code):
        temp_file_name = './data/preprocessed/%s_%s.csv' %(\
            ts_code, \
            self.setting_name_stock_pp)
        return temp_file_name

    def FileNameMergePPData(self):
        return './data/preprocessed/merge_%s.npy' % self.setting_name

    def InvalidFileClean(self, file_name):
        if os.path.exists(file_name):
            temp_df = pd.read_csv(file_name)
            if len(temp_df) == 0:
                os.remove(file_name)
                return True
        return False

    def StockDataMerge(self, df_1, df_2):
        df_2.drop(['ts_code'],axis=1,inplace=True)
        df_merge = pd.merge(df_1, df_2, left_on='trade_date', right_on='trade_date')
        return df_merge


    def TradeDayDataMerge(self, df_1, df_2):
        df_2.drop(['trade_date'],axis=1,inplace=True)
        df_merge = pd.merge(df_1, df_2, left_on='ts_code', right_on='ts_code')
        return df_merge

    def DownloadStockData(self, ts_code):
        start_date = '19000101'
        end_date = self.end_date
        name_list = self.FileNameStockDownloadData(ts_code)
        while True:
            file_name = name_list[0]
            if not os.path.exists(file_name):
                df = pro.daily(ts_code = ts_code, start_date = start_date, end_date = end_date)
                base_common.MKFileDirs(file_name)
                df.to_csv(file_name)
                if self.InvalidFileClean(file_name):
                    continue

            file_name = name_list[1]
            if not os.path.exists(file_name) and self.use_daily_basic:
                df = pro.daily_basic(ts_code = ts_code, start_date = start_date, end_date = end_date)
                base_common.MKFileDirs(file_name)
                df.to_csv(file_name)
                if self.InvalidFileClean(file_name):
                    continue
            
            file_name = name_list[2]
            if not os.path.exists(file_name) and self.use_money_flow:
                df = pro.moneyflow(ts_code = ts_code, start_date = start_date, end_date = end_date)
                base_common.MKFileDirs(file_name)
                df.to_csv(file_name)
                if self.InvalidFileClean(file_name):
                    continue

            file_name = name_list[3]
            if not os.path.exists(file_name) and self.use_adj_factor:
                df = pro.adj_factor(ts_code = ts_code, trade_date='')
                col_date = df['trade_date'].copy()
                col_date = pd.to_numeric(col_date)
                df = df[col_date <= int(end_date)].copy().reset_index(drop=True)
                base_common.MKFileDirs(file_name)
                df.to_csv(file_name)
                if self.InvalidFileClean(file_name):
                    continue
            break

    def DownloadData(self):
        for code_index in range(0, len(self.code_list)):
            ts_code = self.code_list[code_index]
            self.DownloadStockData(ts_code)
            print("%-4d : %s 100%%" % (code_index, ts_code))

    def LoadDownloadStockData(self, ts_code):
        download_name_list = self.FileNameStockDownloadData(ts_code)
        # daily
        if not os.path.exists(download_name_list[0]):
            return pd.DataFrame()
        merge_df = pd.read_csv(download_name_list[0])

        # daily basic
        if self.use_daily_basic:
            if not os.path.exists(download_name_list[1]):
                return pd.DataFrame()
            df_daily_basic = pd.read_csv(download_name_list[1])
            df_daily_basic.drop(['close'],axis=1,inplace=True)
            merge_df = self.StockDataMerge(merge_df, df_daily_basic)

        # money flow
        if self.use_money_flow:
            if not os.path.exists(download_name_list[2]):
                return pd.DataFrame()
            df_money_flow = pd.read_csv(download_name_list[2])
            merge_df = self.StockDataMerge(merge_df, df_money_flow)

        # adj factor
        if self.use_adj_factor:
            if not os.path.exists(download_name_list[3]):
                return pd.DataFrame()
            df_adj_factor = pd.read_csv(download_name_list[3])
            merge_df = self.StockDataMerge(merge_df, df_adj_factor)

        return merge_df


    def UpdateStockPPData(self, ts_code):
        stock_pp_file_name = self.FileNameStockPPData(ts_code)
        if not os.path.exists(stock_pp_file_name):
            ###################### temp transfer ##########################
            temp_setting_name = '%s_%u_%u_%u' % (\
                            str(self.end_date), \
                            int(self.use_daily_basic), \
                            int(self.use_money_flow), \
                            int(self.use_adj_factor))
            temp_file_name = './data/preprocessed/%s_%s.csv' %(\
                            ts_code, \
                            temp_setting_name)
            if os.path.exists(temp_file_name):
                shutil.move(temp_file_name, stock_pp_file_name)
                return
            ###############################################################
            merge_df = self.LoadDownloadStockData(ts_code)
            pp_data = preprocess.StockDataPreProcess(merge_df, self.adj_mode)
            if len(pp_data) > 0:
                base_common.MKFileDirs(stock_pp_file_name)
                pp_data.to_csv(stock_pp_file_name)
            else:
                print("UpdateStockPPData error: %s" % ts_code)
                return
            sys.stdout.write("%-4d : %s 100%%\n" % (self.code_index_map[ts_code], ts_code))

    def LoadStockPPData(self, ts_code, cut_from_start_date=False):
        stock_pp_file_name = self.FileNameStockPPData(ts_code)
        if not os.path.exists(stock_pp_file_name):
            return pd.DataFrame()
        pp_data = pd.read_csv(stock_pp_file_name)
        if cut_from_start_date:
            pp_data = pp_data[pp_data['trade_date'] >= int(self.start_date)]
        return pp_data

    def UpdatePPData(self):
        base_common.ListMultiThread(UpdatePPDataMTFunc, self, 8, self.code_list)
        # for code_index in range(0, len(self.code_list)):
        #     ts_code = self.code_list[code_index]
        #     self.UpdateStockPPData(ts_code)
        #     print("%-4d : %s 100%%" % (code_index, ts_code))


    def DownloadTradeDayData(self, trade_date):
        name_list = self.FileNameTradeDayDownloadData(trade_date)
        while True:
            file_name = name_list[0]
            if not os.path.exists(file_name):
                df = pro.daily(trade_date = trade_date)
                base_common.MKFileDirs(file_name)
                df.to_csv(file_name)
                if self.InvalidFileClean(file_name):
                    continue

            file_name = name_list[1]
            if not os.path.exists(file_name) and self.use_daily_basic:
                df = pro.daily_basic(trade_date = trade_date)
                base_common.MKFileDirs(file_name)
                df.to_csv(file_name)
                if self.InvalidFileClean(file_name):
                    continue
            
            file_name = name_list[2]
            if not os.path.exists(file_name) and self.use_money_flow:
                df = pro.moneyflow(trade_date = trade_date)
                base_common.MKFileDirs(file_name)
                df.to_csv(file_name)
                if self.InvalidFileClean(file_name):
                    continue

            file_name = name_list[3]
            if not os.path.exists(file_name) and self.use_adj_factor:
                df = pro.adj_factor(ts_code='', trade_date=trade_date)
                base_common.MKFileDirs(file_name)
                df.to_csv(file_name)
                if self.InvalidFileClean(file_name):
                    continue
            break

    def LoadDownloadTradeDayData(self, trade_date):
        download_name_list = self.FileNameTradeDayDownloadData(trade_date)
        # daily
        if not os.path.exists(download_name_list[0]):
            return pd.DataFrame()
        merge_df = pd.read_csv(download_name_list[0])

        # daily basic
        if self.use_daily_basic:
            if not os.path.exists(download_name_list[1]):
                return pd.DataFrame()
            df_daily_basic = pd.read_csv(download_name_list[1])
            df_daily_basic.drop(['close'],axis=1,inplace=True)
            merge_df = self.TradeDayDataMerge(merge_df, df_daily_basic)

        # money flow
        if self.use_money_flow:
            if not os.path.exists(download_name_list[2]):
                return
            df_money_flow = pd.read_csv(download_name_list[2])
            merge_df = self.TradeDayDataMerge(merge_df, df_money_flow)

        # adj factor
        if self.use_adj_factor:
            if not os.path.exists(download_name_list[3]):
                return
            df_adj_factor = pd.read_csv(download_name_list[3])
            merge_df = self.TradeDayDataMerge(merge_df, df_adj_factor)

        return merge_df


if __name__ == "__main__":
    data_source = DataSource(20000101, '', '', 1, 20000101, 20200106, False, False, True)
    data_source.ShowStockCodes()
    data_source.DownloadData()
    data_source.UpdatePPData()

