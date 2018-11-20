# -*- coding:UTF-8 -*-

import tensorflow as tf
from tensorflow import keras
import tushare as ts
import numpy as np
import pandas as pd
import os
import time
import sys

reload(sys)
sys.setdefaultencoding('utf-8')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

#pd.set_option('display.width', 150)  # 设置字符显示宽度
#pd.set_option('display.max_rows', None)  # 设置显示最大行

print("load...")
train_data=np.load("train_data.npy")
print("train_data: {}".format(train_data.shape))

print("reorder...")
order=np.argsort(np.random.random(len(train_data)))
train_data=train_data[order]

col_num=train_data.shape[1]
train_features=train_data[:,0:col_num-1]
train_labels=train_data[:,col_num-1:]
print("train_features: {}".format(train_features.shape))
print("train_labels: {}".format(train_labels.shape))

mean = train_features.mean(axis=0)
std = train_features.std(axis=0)
print("mean: {}".format(mean.shape))
print(mean)
print("std: {}".format(std.shape))
print(std)

np.save('mean.npy', mean)
np.save('std.npy', std)

temp=np.load('mean.npy')
print(temp)
temp=np.load('std.npy')
print(temp)