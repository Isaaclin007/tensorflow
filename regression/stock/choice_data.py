# -*- coding:UTF-8 -*-

import tensorflow as tf
from tensorflow import keras
import tushare as ts
import numpy as np
import pandas as pd
import os
import time
import sys
from EmQuantAPI import *

login = c.start("forcelogin = 1")
data=c.csd("00001.HK", \
    "OPEN,CLOSE,HIGH,LOW,VOLUME,AMOUNT,TURN", \
    "1980-01-01", \
    "2019-02-21", \
    "period=1,adjustflag=1,curtype=1,pricetype=1,order=1,market=CNSESH,ispandas=1")
data.dropna(axis = 0)
print(data)

# data.to_csv('./temp.csv')

# for code in data.Codes:
#     for i in range(0, len(data.Indicators)):
#         for j in range(0, len(data.Dates)):
#             print data.Data[code][i][j]
