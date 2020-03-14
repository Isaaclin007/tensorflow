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

def DecayTest():
    label_days = 6
    show_data_1 = np.zeros((label_days, 2))
    y = 1.0
    cnt = 0
    while 1:
        show_data_1[cnt][0] = cnt
        show_data_1[cnt][1] = y
        y *= 0.6
        cnt += 1
        if cnt == label_days:
            break
    print(show_data_1)

    show_data_2 = np.zeros((label_days, 2))
    y = 1.0
    cnt = 0
    while 1:
        show_data_2[cnt][0] = cnt
        show_data_2[cnt][1] = y
        y -= 0.185
        cnt += 1
        if cnt == label_days:
            break
    print(show_data_2)
    np_common.Show2DData('Decay', [show_data_1, show_data_2], [])


DecayTest()