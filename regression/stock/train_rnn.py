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

reload(sys)
sys.setdefaultencoding('utf-8')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

def mystockloss(y_true, y_pred, e=0.1):
    return abs(y_true - y_pred) / 10.0 * K.max([y_true, y_pred, y_true*0.0])

def build_model(input_layer_shape):
    HIDDEN_SIZE = 32

    model = keras.models.Sequential()
    # model.add(keras.layers.SimpleRNN(HIDDEN_SIZE, return_sequences=False,
    #                     input_shape=(input_layer_shape),unroll=True))
    model.add(keras.layers.LSTM(32, input_shape=(input_layer_shape), return_sequences=True))
    # model.add(keras.layers.LSTM(32, return_sequences=True))
    model.add(keras.layers.LSTM(10))
    model.add(keras.layers.Dense(1))
    model.compile(loss="mae", optimizer="rmsprop")
    # my_optimizer = tf.train.RMSPropOptimizer(0.002)
    # model.compile(loss=mystockloss,
    #                 optimizer=my_optimizer,
    #                 metrics=[mystockloss])
    return model

def train():
    # train_features, train_labels = tushare_data.GetTrainData()
    train_features, train_labels = wave_kernel.GetTrainDataOriginal()
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

    EPOCHS = 500
    BATCH_SIZE = 128

    # Display training progress by printing a single dot for each completed epoch.
    class PrintDot(keras.callbacks.Callback):
        def on_epoch_end(self,epoch,logs):
            if epoch % 1 == 0: 
                sys.stdout.write('\r%d' % (epoch))
                sys.stdout.flush()
            #print('.', end='')
            #print('.')

    # The patience parameter is the amount of epochs to check for improvement.
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

    where_are_nan = np.isnan(train_features)
    where_are_inf = np.isinf(train_features)
    train_features[where_are_nan] = 0.0
    train_features[where_are_inf] = 0.0
    history = model.fit(train_features, train_labels, epochs=EPOCHS, batch_size=BATCH_SIZE, 
                        validation_split=0.1, verbose=0,
                        # callbacks=[early_stop, PrintDot()])
                        callbacks=[PrintDot()])


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

    model.save("./model/model.h5")
    np.save('./model/mean.npy', mean)
    np.save('./model/std.npy', std)

if __name__ == "__main__":
    train()
    # lossTest()
