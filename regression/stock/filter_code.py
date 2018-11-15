# -*- coding:UTF-8 -*-


import tushare as ts
import numpy as np
import pandas as pd
import os
import time
import sys

reload(sys)
sys.setdefaultencoding('utf-8')

pd.set_option('display.width', 150)  # 设置字符显示宽度
#pd.set_option('display.max_rows', None)  # 设置显示最大行

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

print("load_df.dtypes:")
print(load_df.dtypes)
print("\n\n\n")

print(load_df)
print("\n\n\n")

pd.set_option('display.max_rows', 10)  # 设置显示最大行
load_df=load_df[load_df['list_date']<=20100101]
load_df=load_df[load_df['industry']=='软件服务']
print(load_df)
print("\n\n\n")

code_list=load_df['ts_code'].values
# print("len(load_df):%d" % len(load_df))
# for iloop in range(0,len(load_df)):
#     #code_list.append(load_df['ts_code'][iloop])
#     print("iloop:%d" % iloop)
#     temp=load_df['list_date'][iloop]
#     #print(load_df['list_date'][iloop])
print('code_list\n')
print(code_list)
print("\n\n\n")
