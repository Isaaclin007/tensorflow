# -*- coding:UTF-8 -*-


import tushare as ts
import numpy as np
import pandas as pd
import os
import time
import datetime
import sys
import math
sys.path.append("..")
from common import base_common
from common import np_common
from common.const_def import *

# feature unit type 特征单元常量定义 NORM:数据归一化，计算与 avg100 的比例
FUT_D5_NORM = 1  # 每天的5个数据归一化
FUT_5REGION5_NORM = 2  # 每天的5个 5日区间 数据归一化
FUT_2AVG5_NORM = 3  # 每天的2个 avg5 数据归一化

# acture data index 定义
ADI_OPEN_INCREASE = 0
ADI_LOW_INCREASE = 1
ADI_OPEN = 2
ADI_LOW = 3
ADI_CLOSE = 4
ADI_TSCODE = 5
ADI_DATE = 6

INVALID_DATE = 0.0

class Feature():
    def __init__(self, 
                 feature_unit_num, 
                 feature_unit_type, 
                 feature_unit_step,
                 filter_suspend_data, 
                 filter_adj_flag_data):
        self.feature_unit_num = feature_unit_num
        self.feature_unit_type = feature_unit_type
        self.feature_unit_step = feature_unit_step
        self.feature_ref_days_num = feature_unit_num * feature_unit_step
        self.filter_suspend_data = filter_suspend_data
        self.filter_adj_flag_data = filter_adj_flag_data
        self.UpdateFeatureSize()
        self.setting_name = '%u_%u_%u_%u_%u' % (feature_unit_num, 
                                             feature_unit_type, 
                                             feature_unit_step,
                                             int(filter_suspend_data), 
                                             int(filter_adj_flag_data))

    def UpdateFeatureSize(self):
        if self.feature_unit_type == FUT_D5_NORM:
            self.feature_unit_size = 5
        elif self.feature_unit_type == FUT_5REGION5_NORM:
            self.feature_unit_size = 5
        elif self.feature_unit_type == FUT_2AVG5_NORM:
            self.feature_unit_size = 2
        self.feature_size = self.feature_unit_size * self.feature_unit_num
        self.acture_size = 7
        self.index_open_increase = self.feature_size + ADI_OPEN_INCREASE
        self.index_low_increase = self.feature_size + ADI_LOW_INCREASE
        self.index_open = self.feature_size + ADI_OPEN
        self.index_low = self.feature_size + ADI_LOW
        self.index_close = self.feature_size + ADI_CLOSE
        self.index_tscode = self.feature_size + ADI_TSCODE
        self.index_date = self.feature_size + ADI_DATE

        self.unit_size = self.feature_size + self.acture_size

    def AppendFeature_Filter(self, pp_data, day_index):
        data_len = len(pp_data)
        if (day_index + self.feature_ref_days_num) > data_len:
            # print('Error: AppendFeature_Filter day_index=%d, data_len=%u' % (day_index, data_len))
            return False
        if self.filter_suspend_data or self.filter_adj_flag_data:
            for iloop in range(day_index, day_index + self.feature_ref_days_num):
                if self.filter_suspend_data:
                    if pp_data[iloop][PPI_suspend] != 0:
                        return False
                # if self.filter_adj_flag_data:
                #     if pp_data['adj_flag'][iloop] != 0:
                #         return False
        return True

    def AppendFeature_FUT_D5_NORM(self, pp_data, day_index, data_unit):
        if not self.AppendFeature_Filter(pp_data, day_index):
            return False
        base_close = pp_data[day_index][PPI_close_100_avg]
        base_vol = pp_data[day_index][PPI_vol_100_avg]
        for iloop in reversed(range(self.feature_unit_num)):
            temp_index = day_index + iloop * self.feature_unit_step
            data_unit.append(pp_data[temp_index][PPI_open] / base_close)
            data_unit.append(pp_data[temp_index][PPI_close] / base_close)
            data_unit.append(pp_data[temp_index][PPI_high] / base_close)
            data_unit.append(pp_data[temp_index][PPI_low] / base_close)
            data_unit.append(pp_data[temp_index][PPI_vol] / base_vol)
        return True

    def AppendFeature_FUT_5REGION5_NORM(self, pp_data, day_index, data_unit):
        if not self.AppendFeature_Filter(pp_data, day_index):
            return False
        base_close = pp_data[day_index][PPI_close_100_avg]
        base_vol = pp_data[day_index][PPI_vol_100_avg]
        for iloop in reversed(range(self.feature_unit_num)):
            temp_index = day_index + iloop * self.feature_unit_step
            data_unit.append(pp_data[temp_index][PPI_open_5] / base_close)
            data_unit.append(pp_data[temp_index][PPI_close_5] / base_close)
            data_unit.append(pp_data[temp_index][PPI_high_5] / base_close)
            data_unit.append(pp_data[temp_index][PPI_low_5] / base_close)
            data_unit.append(pp_data[temp_index][PPI_vol_5] / base_vol)
        return True

    def AppendFeature_FUT_2AVG5_NORM(self, pp_data, day_index, data_unit):
        if not self.AppendFeature_Filter(pp_data, day_index):
            return False
        base_close = pp_data[day_index][PPI_close_100_avg]
        base_vol = pp_data[day_index][PPI_vol_100_avg]
        for iloop in reversed(range(self.feature_unit_num)):
            temp_index = day_index + iloop * self.feature_unit_step
            data_unit.append(pp_data[temp_index][PPI_close_5_avg] / base_close)
            data_unit.append(pp_data[temp_index][PPI_vol_5_avg] / base_vol)
        return True

    # 从 day_index 开始的前 feature_unit_num 组的数据，包含 day_index
    def AppendFeature(self, pp_data, day_index, data_unit):
        data_len = len(pp_data)
        if self.feature_unit_type == FUT_D5_NORM:
            return self.AppendFeature_FUT_D5_NORM(pp_data, day_index, data_unit)
        elif self.feature_unit_type == FUT_5REGION5_NORM:
            return self.AppendFeature_FUT_5REGION5_NORM(pp_data, day_index, data_unit)
        elif self.feature_unit_type == FUT_2AVG5_NORM:
            return self.AppendFeature_FUT_2AVG5_NORM(pp_data, day_index, data_unit)
        else:
            return False

    #  day_index 当天的数据
    def AppendActureData(self, pp_data, day_index, data_unit):
        if day_index >= len(pp_data):
            return False
        if day_index < 0:
            return False
        data_unit.append(pp_data[day_index][PPI_open_increase])
        data_unit.append(pp_data[day_index][PPI_low_increase])
        data_unit.append(pp_data[day_index][PPI_open])
        data_unit.append(pp_data[day_index][PPI_low])
        data_unit.append(pp_data[day_index][PPI_close])
        data_unit.append(pp_data[day_index][PPI_ts_code])
        data_unit.append(pp_data[day_index][PPI_trade_date])
        return True

    def GetDataUnit(self, pp_data, day_index):
        data_unit=[]
        if not self.AppendFeature(pp_data, day_index, data_unit):
            return []
        if not self.AppendActureData(pp_data, day_index, data_unit):
            return []
        return data_unit


if __name__ == "__main__":
    print('features.py')

