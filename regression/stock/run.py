# -*- coding:UTF-8 -*-

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
from tensorflow import keras
from compiler.ast import flatten
import numpy as np
import pandas as pd
import time
import sys
import tushare_data
import train
import test


tushare_data.UpdateTrainTestData()
print("\n")

test_capital_increase = []
for iloop in range(0, 1):
    train.train()
    test_capital_increase.append(test.TestEntry(0, 1, False))
print('\ntest_capital_increase:')
for iloop in range(0, len(test_capital_increase)):
    print('capital_increase[%d]: %.2f' % (iloop, test_capital_increase[iloop]))

print('\nmean: %.2f' % np.mean(test_capital_increase))

print('\nvar: %.2f' % np.var(test_capital_increase))


