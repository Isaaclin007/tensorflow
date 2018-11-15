# -*- coding:UTF-8 -*-


import tushare as ts
import numpy as np
import pandas as pd
import os
import time
print(ts.__version__)
ts.set_token('230c446ae448ec95357d0f7e804ddeebc7a51ff340b4e6e0913ea2fa')
pro = ts.pro_api()

#下载数据，生成feature_list和label_list  <<<<<<
feature_list=[]
label_list=[]

code_list=['600050.SH', '600104.SH']
current_date=time.strftime('%Y%m%d',time.localtime(time.time()))
for code_index in range(0, len(code_list)):
    stock_code=code_list[code_index]
    file_name='./data/'+stock_code+'_'+current_date+'.csv'
    if os.path.exists(file_name):
        print("read_csv:%s" % file_name)
        load_df=pd.read_csv(file_name)
    else:
        load_df=pro.daily_basic(ts_code=stock_code, start_date='20100101', end_date=current_date)
        load_df.to_csv(file_name)

    src_df=load_df[['trade_date', 'close', 'turnover_rate']].copy()
    src_df['increase']=0.0

    for iloop in range(0, len(src_df)-1):
        src_df.iloc[iloop,3]=(100*src_df['close'][iloop]/src_df['close'][iloop+1])-100.0
        #src_df['increase'][iloop]=(src_df['close'][iloop]/src_df['close'][iloop-1])-1.0
    print("src_df: {}".format(src_df.shape))
    print(src_df)

    feature_days=10
    for day_loop in range(0, len(src_df)-feature_days-1):
        feature=[]
        for iloop in range(0, feature_days):
            feature.append(src_df['increase'][day_loop+iloop+1])
            feature.append(src_df['turnover_rate'][day_loop+iloop+1])
        feature_list.append(feature)
        label_list.append(src_df['increase'][day_loop])
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

print("\n\n\n")

feature_data=np.array(feature_list)
label_data=np.array(label_list)
print("feature_data.shape: {}".format(feature_data.shape))
print("label_data.shape: {}".format(label_data.shape))

print("feature_data[0]:")
print(feature_data[0])
print("feature_data[1]:")
print(feature_data[1])
print("\n\n\n")

print("label_data:")
print(label_data)
print("\n\n\n")


#np_arr=np.array(temp_data)
#print(np_arr)


#print("len(temp_data): ")
#print(len(temp_data))
#print(temp_data)

#feature_data.shape= (40,1)
#print(feature_data)
