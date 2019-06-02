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
import wave_test
import wave_test_regression
import wave_kernel

tushare_data.UpdatePreprocessData()
wave_test.TestAllStocks()


