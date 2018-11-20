# -*- coding:UTF-8 -*-

import tensorflow as tf
from tensorflow import keras
import tushare as ts
import numpy as np
import pandas as pd
import os
import time
import sys
import datetime

reload(sys)
sys.setdefaultencoding('utf-8')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

print(ts.__version__)
ts.set_token('230c446ae448ec95357d0f7e804ddeebc7a51ff340b4e6e0913ea2fa')
pro = ts.pro_api()

current_date=time.strftime('%Y%m%d',time.localtime(time.time()))
file_name='./data/'+current_date+'.csv'
if os.path.exists(file_name):
    load_df=pd.read_csv(file_name)
else:
    load_df=pro.daily_basic(ts_code='', trade_date=current_date)
    load_df.to_csv(file_name)
print("load_df set: {}".format(load_df.shape))

result_sort=pd.read_csv('result_sort.csv')
result_sort['acture_increase']=0.0
print(result_sort)