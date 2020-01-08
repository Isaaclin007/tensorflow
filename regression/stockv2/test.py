# -*- coding:UTF-8 -*-


import tushare as ts
import numpy as np
import pandas as pd
import os
import time
import datetime
import sys
import math
import preprocess
sys.path.append("..")
from common import base_common
from common import np_common


print(np.random.random(10))

# base_common.MKFileDirs('./a/c/c/c')

# def GetFeature():
#     data_unit = []
#     for iloop in range(150):
#         data_unit.append(iloop)
#     return data_unit

# L = 0
# def SetFeature():
#     # data_unit = []
#     for iloop in range(150):
#         # data_unit.append(iloop)
#         L = iloop
#     # return L

# start = time.time()
# for iloop in range(1000000):
#     SetFeature()
# print(time.time() - start)