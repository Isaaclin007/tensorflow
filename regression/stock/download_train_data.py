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


tushare_data.DownloadTrainData()
print("\n\n\n")


