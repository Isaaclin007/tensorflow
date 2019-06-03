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

reload(sys)
sys.setdefaultencoding('utf-8')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

#pd.set_option('display.width', 150)  # 设置字符显示宽度
#pd.set_option('display.max_rows', None)  # 设置显示最大行

# print("load...")
# # train_data=np.load("./temp_data/train_data.npy")
# train_data=np.load("../stock_/train_data.npy")
# print("train_data: {}".format(train_data.shape))

# print("reorder...")
# order=np.argsort(np.random.random(len(train_data)))
# train_data=train_data[order]
# train_data=train_data[:2000000]

# feature_size=tushare_data.FeatureSize()
# label_index=tushare_data.LabelColIndex()
# # train_features=train_data[:,0:feature_size]
# # train_labels=train_data[:,label_index:label_index+1]
# train_features=train_data[:,0:85]
# train_labels=train_data[:,85:86]
# print("train_features: {}".format(train_features.shape))
# print("train_labels: {}".format(train_labels.shape))

#Create the model
def myloss(y_true, y_pred, e=0.1):
    return (1-e)*((y_true * y_true - y_pred * y_pred) * (y_true * y_true - y_pred * y_pred))

def mymseloss(y_true, y_pred, e=0.1):
    return (1-e)*((y_true - y_pred) * (y_true - y_pred))

def mymaeloss(y_true, y_pred, e=0.1):
    return (1-e)* abs(y_true - y_pred)

def mystockloss(y_true, y_pred, e=0.1):
    # return (1-e) * abs(y_true - y_pred) * K.max([K.max([y_true, y_pred]) - 0.0, 0.0])
    # return (1-e) * abs(y_true - y_pred) * K.pow(1.2, K.max([y_true, y_pred])) / 100.0
    # max_y = K.max([y_true, y_pred]) / 10.0  # predict_day_count
    # return (1-e) * abs(y_true - y_pred) * tf.where(tf.greater(max_y, 5.0), max_y / 5.0, K.pow(1.584893, max_y) / 10.0)

    # max_y = K.max([y_true, y_pred]) / 10.0  # predict_day_count
    # return (1-e) * abs(y_true - y_pred) * K.pow(2.0, max_y) / 10.0
    return abs(y_true - y_pred) / 10.0 * K.max([y_true, y_pred, y_true*0.0])

    # return (1-e) * abs(y_true - y_pred) * K.pow(1.1, y_true)

def lossValue(y_true, y_pred):
    temp_loss = mystockloss(y_true, y_pred)
    with tf.Session() as sess:
        return temp_loss.eval()

def printLoss(y_true, y_pred):
    print('%-20f%-20f%-20f' % (y_true, y_pred, lossValue(y_true, y_pred)))

def lossTest():
    print('%-20s%-20s%-20s' % ('y_true', 'y_pred', 'loss'))
    print('----------------------------------------------')

    printLoss(-20.0, -19.0)
    printLoss(-20.0, 20.0)
    printLoss(-20.0, 0.0)
    printLoss(20.0, 0.0)
    printLoss(50.0, 0.0)
    printLoss(50.0, 49.0)
    printLoss(100.0, 0.0)
    printLoss(100.0, 99.0)
    printLoss(-100.0, 0.0)
    printLoss(-100.0, 0.0)

def build_model(input_layer_shape):
    model = keras.Sequential([
        keras.layers.Dense(8, activation=tf.nn.relu, input_shape=input_layer_shape),
        # keras.layers.Dense(128, activation=tf.nn.relu),
        # keras.layers.Dense(128, activation=tf.nn.relu),
        # keras.layers.Dense(128, activation=tf.nn.relu),
        # keras.layers.Dense(128, activation=tf.nn.relu),
        # keras.layers.Dense(128, activation=tf.nn.relu),
        # keras.layers.Dense(128, activation=tf.nn.relu),
        # keras.layers.Dense(128, activation=tf.nn.relu),
        # keras.layers.Dense(128, activation=tf.nn.relu),
        # keras.layers.Dense(128, activation=tf.nn.relu),
        # keras.layers.Dense(128, activation=tf.nn.relu),
        # keras.layers.Dense(128, activation=tf.nn.relu),
        # keras.layers.Dense(64, activation=tf.nn.relu),
        # keras.layers.Dense(32, activation=tf.nn.relu),
        # keras.layers.Dense(8, activation=tf.nn.relu),
        keras.layers.Dense(4, activation=tf.nn.relu),
        keras.layers.Dense(1)
    ])

    optimizer = tf.train.RMSPropOptimizer(0.01)

    # model.compile(loss='mse',
    #                 optimizer=optimizer,
    #                 metrics=['mae'])
    # model.compile(loss='mae',
    #                 optimizer=optimizer,
    #                 metrics=['mae'])
    model.compile(loss=mystockloss,
                    optimizer=optimizer,
                    metrics=[mystockloss])
                    # metrics=['mae'])
    return model

def train():
    train_features, train_labels = wave_kernel.GetTrainData()
    # train_features = tushare_data.Features10D14To10D5(train_features)
    print("train_features: {}".format(train_features.shape))

    mean = train_features.mean(axis=0)
    std = train_features.std(axis=0)
    # print("mean: {}".format(mean.shape))
    # print("std: {}".format(std.shape))
    train_features = (train_features - mean) / std

    # max_label_value = tushare_data.predict_day_count * 10.0
    # temp_mask = train_labels >= max_label_value
    # train_labels[temp_mask] = max_label_value

    model = build_model((train_features.shape[1],))
    model.summary()

    EPOCHS = 100

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
    history = model.fit(train_features, train_labels, epochs=EPOCHS,
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
