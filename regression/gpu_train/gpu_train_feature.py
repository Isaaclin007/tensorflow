# -*- coding:UTF-8 -*-


import numpy as np
import os
import time
import datetime
import sys
import math

# 常量定义
FEATURE_G0_D5_AVG = 0
LABEL_PRE_CLOSE_2_TD_CLOSE = 0
LABEL_T1_OPEN_2_TD_CLOSE = 1
LABEL_CONSECUTIVE_RISE_SCORE = 2
LABEL_T1_OPEN_2_TD_OPEN = 3
INVALID_DATE = 99990101.0

# 设置参数
feature_days = 30
label_days = 10
active_label_day = 1  # 0 ~ (label_days-1)
feature_type = FEATURE_G0_D5_AVG
label_type = LABEL_T1_OPEN_2_TD_CLOSE

# 计算参数
feature_unit_size = 0
if feature_type == FEATURE_G0_D5_AVG:
    feature_unit_size = 5
feature_size = feature_unit_size * feature_days
acture_unit_size = 7

ACTURE_DATA_INDEX_OPEN_INCREASE = 0
ACTURE_DATA_INDEX_LOW_INCREASE = 1
ACTURE_DATA_INDEX_OPEN = 2
ACTURE_DATA_INDEX_LOW = 3
ACTURE_DATA_INDEX_CLOSE = 4
ACTURE_DATA_INDEX_TSCODE = 5
ACTURE_DATA_INDEX_DATE = 6

def ACTIVE_LABEL_DAY():
    return active_label_day

def COL_ACTIVE_LABEL():
    return feature_size + active_label_day

def COL_FEATURE_OFFSET():
    return 0

def FEATURE_SIZE():
    return feature_size

def ACTURE_UNIT_SIZE():
    return acture_unit_size

def COL_ACTURE_OFFSET(actrue_day_index):
    return (feature_size + label_days + (actrue_day_index * acture_unit_size))

def COL_TRADE_DATE(actrue_day_index):
    return (COL_ACTURE_OFFSET(actrue_day_index) + ACTURE_DATA_INDEX_DATE)

def SettingName():
    return ('%u_%u_%u_%u' % (feature_type, label_type, feature_days, active_label_day))


if __name__ == "__main__":
    print('features.py')

