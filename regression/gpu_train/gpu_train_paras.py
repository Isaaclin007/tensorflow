# -*- coding:UTF-8 -*-

import numpy as np
import pandas as pd
import os
import time
import sys
import math
import getopt


train_data_ = 'fix'  
    # fix 
    # wave
data_split_mode_ = 'splitbydate'
    # samplebydate
    # splitbydate
    # random
val_split_ = 0.2
model_type_ = 'LSTM'
    # LSTM 
    # Dense
lstm_size_ = 64
lstm_dense_size_ = 1
dense_layer_num_ = 1
dense_size_ = [4, 4, 4, 4]
loss_func_ = 'LossTanhDiff'  
    # LossTP0MaxRatio 
    # LossT10P0MaxRatio 
    # LossT2P0MaxRatio 
    # LossTP1MaxRatio 
    # LossTs5Ps50MaxRatio 
    # LossTs5Ps50MaxRatioMean 
    # LossTs9Ps90MaxRatio 
    # LossAbs 
    # LossTP010ClipDiff 
    # LossTanhDiff
optimizer_ = 'KerasRMSProp'  
    # RMSProp 
    # KerasRMSProp

learning_rate_ = 0.004
batch_size_ = 10240
epochs_ = 500
use_test_data_ = True




def InitParas(argv):
    global train_data_
    global data_split_mode_
    global val_split_
    global model_type_
    global lstm_size_
    global lstm_dense_size_
    global dense_size_
    global dense_layer_num_
    global loss_func_
    global optimizer_
    global learning_rate_
    global batch_size_
    global epochs_

    opts,args = getopt.getopt(argv[1:],'-h-v', ['help',
                                                'version',
                                                'train_data=',
                                                'data_split_mode=',
                                                'val_split=',
                                                'model_type=',
                                                'lstm_size=',
                                                'lstm_dense_size=',
                                                'dense_size=',
                                                'loss_func=',
                                                'optimizer=',
                                                'fix_activer_label_day=',
                                                'learning_rate=',
                                                'batch_size=',
                                                'epochs='
                                                ])
    # if len(args) > 0:
    #     print('InitParas.Error, unsupported args:')
    #     print(args)
    #     return False
    for opt_name,opt_value in opts:
        print("%s:%s" % (opt_name, opt_value))
        if opt_name == '--train_data':
            train_data_ = opt_value
        elif opt_name == '--data_split_mode':
            data_split_mode_ = opt_value
        elif opt_name == '--val_split':
            val_split_ = float(opt_value)
        elif opt_name == '--model_type':
            model_type_ = opt_value
        elif opt_name == '--lstm_size':
            lstm_size_ = int(opt_value)
        elif opt_name == '--lstm_dense_size':
            lstm_dense_size_ = int(opt_value)
        elif opt_name == '--dense_size':
            dense_size_ = map(int, opt_value.split(','))
            dense_layer_num_ = len(dense_size_)
        elif opt_name == '--loss_func':
            loss_func_ = opt_value
        elif opt_name == '--optimizer':
            optimizer_ = opt_value
        elif opt_name == '--learning_rate':
            learning_rate_ = float(opt_value)
        elif opt_name == '--batch_size':
            batch_size_ = int(opt_value)
        elif opt_name == '--epochs':
            epochs_ = int(opt_value)
        # else:
        #     print('InitParas.Error, unsupported opt_name(%s)!' % opt_name)
        #     return False
    return True

def SettingName():
    if model_type_ == 'LSTM':
        model_str = '%s_%u.%u' % (model_type_, lstm_size_, lstm_dense_size_)
    elif model_type_ == 'Dense':
        model_str = '%s_%u' % (model_type_, dense_size_[0])
        for iloop in range(1, dense_layer_num_):
            model_str = model_str + '.%u' % dense_size_[iloop]
    model_setting = '%s_%s_%s_%s_%f_%f_%u_%u' % (model_str,   
                                                loss_func_, 
                                                optimizer_, 
                                                data_split_mode_, 
                                                val_split_, 
                                                learning_rate_, 
                                                batch_size_, 
                                                epochs_
                                                )
    return model_setting

