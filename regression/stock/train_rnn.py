# -*- coding:UTF-8 -*-

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import backend as K
import tushare as ts
import numpy as np
import pandas as pd
import os
import time
import sys
import tushare_data
import math
import wave_kernel
from tensorflow.python.keras.callbacks import LearningRateScheduler

HIDDEN_SIZE = 36
BATCH_SIZE = 10240
LEANING_RATE = 0.004

reload(sys)
sys.setdefaultencoding('utf-8')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

def mystockloss(y_true, y_pred, e=0.1):
    return abs(y_true - y_pred) / 10.0 * K.max([y_true, y_pred, y_true*0.0])

def build_model(input_layer_shape):

    model = keras.models.Sequential()
    # model.add(keras.layers.Dense(HIDDEN_SIZE, input_shape=(input_layer_shape)))
    # model.add(keras.layers.SimpleRNN(HIDDEN_SIZE, return_sequences=False, input_shape=(input_layer_shape),unroll=True))
    model.add(keras.layers.LSTM(HIDDEN_SIZE, input_shape=(input_layer_shape), return_sequences=False))
    # model.add(keras.layers.LSTM(32, input_shape=(input_layer_shape), return_sequences=True))
    # model.add(keras.layers.LSTM(32, return_sequences=True))
    # model.add(keras.layers.LSTM(10))
    model.add(keras.layers.Dense(1))
    # model.compile(loss="mae", optimizer="rmsprop")
    my_optimizer = tf.train.RMSPropOptimizer(LEANING_RATE)
    model.compile(loss=mystockloss,
                    optimizer=my_optimizer,
                    metrics=[mystockloss])
    return model

def ModelFileNames(train_mode):
    if train_mode == "fix":
        temp_path_name = "./model/fix/%s_%u_%u_%f" % (tushare_data.SettingName(), HIDDEN_SIZE, BATCH_SIZE, LEANING_RATE)
    else:
        temp_path_name = "./model/wave/%s_%u_%u_%f" % (tushare_data.SettingName(), HIDDEN_SIZE, BATCH_SIZE, LEANING_RATE)
    model_name = "%s/model.h5" % temp_path_name
    mean_name = "%s/mean.npy" % temp_path_name
    std_name = "%s/std.npy" % temp_path_name
    return temp_path_name, model_name, mean_name, std_name

def SaveModel(train_mode, model, mean, std):
    temp_path_name, model_name, mean_name, std_name = ModelFileNames(train_mode)
    if not os.path.exists(temp_path_name):
        os.makedirs(temp_path_name)
    model.save(model_name)
    np.save(mean_name, mean)
    np.save(std_name, std)

def LoadModel(train_mode):
    temp_path_name, model_name, mean_name, std_name = ModelFileNames(train_mode)
    model = keras.models.load_model(model_name)
    mean = np.load(mean_name)
    std = np.load(std_name)
    return model, mean, std

def train(train_mode):
    if train_mode == "fix":
        train_features, train_labels = tushare_data.GetTrainData()
    else:
        train_features, train_labels = wave_kernel.GetTrainData()
    # train_features = tushare_data.Features10D14To10D5(train_features)
    print("train_features: {}".format(train_features.shape))

    mean = train_features.mean(axis=0)
    std = train_features.std(axis=0)
    # print("mean: {}".format(mean.shape))
    # print("std: {}".format(std.shape))
    train_features = (train_features - mean) / std
    train_features = tushare_data.ReshapeRnnFeatures(train_features)
    print("train_features: {}".format(train_features.shape))

    # max_label_value = tushare_data.predict_day_count * 10.0
    # temp_mask = train_labels >= max_label_value
    # train_labels[temp_mask] = max_label_value

    model = build_model(train_features.shape[1:])
    model.summary()

    EPOCHS = 1000

    # Display training progress by printing a single dot for each completed epoch.
    class PrintDot(keras.callbacks.Callback):
        def on_epoch_end(self,epoch,logs):
            if epoch % 1 == 0: 
                sys.stdout.write('\r%d' % (epoch))
                sys.stdout.flush()
            #print('.', end='')
            #print('.')

    # The patience parameter is the amount of epochs to check for improvement.
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

    where_are_nan = np.isnan(train_features)
    where_are_inf = np.isinf(train_features)
    train_features[where_are_nan] = 0.0
    train_features[where_are_inf] = 0.0
    history = model.fit(train_features, train_labels, epochs=EPOCHS, batch_size=BATCH_SIZE, 
                        validation_split=0.1, verbose=0,
                        callbacks=[early_stop, PrintDot()])
                        # callbacks=[PrintDot()])


    # print("%-12s%-12s%-12s" %('epoch', 'train_err', 'val_err'))
    # for iloop in history.epoch:
    #     train_err=history.history['mean_absolute_error'][iloop]
    #     val_err=history.history['val_mean_absolute_error'][iloop]
    #     print("%8u%8.2f%8.2f" %(iloop, train_err, val_err))

    # # 显示 <<<<<<<<<<
    import matplotlib.pyplot as plt
    def plot_history(history):
        plt.figure()
        plt.xlabel('Epoch')
        plt.ylabel('loss')
        plt.plot(history.epoch, np.array(history.history['loss']), 
                label='Train Loss')
        plt.plot(history.epoch, np.array(history.history['val_loss']),
                label = 'Val loss')
        plt.legend()
        #plt.ylim([0,5])
        plt.show()
    print("\nplot_history")
    plot_history(history)
    # # 显示 >>>>>>>>>>>>

    SaveModel(train_mode, model, mean, std)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        train(sys.argv[1])
    else:
        train("");
    # lossTest()
