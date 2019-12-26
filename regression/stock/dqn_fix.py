# -*- coding:UTF-8 -*-


import tushare as ts
import numpy as np
import pandas as pd
import os
import time
import datetime
import sys
import math
import random
from absl import app
from absl import flags
import tushare_data
import feature
import pp_daily_update
import dqn_dataset
from collections import deque
from tensorflow import keras
from tensorflow.python.keras import backend as K
import matplotlib.pyplot as plt
import dqn
import dqn_fix_dataset

FLAGS = flags.FLAGS
flags.DEFINE_boolean('test', False, 'test mode')

EPOCHS = 1000
BATCH_SIZE = 10240
LSTM_SIZE = 16
use_test_data = False
loss_func_ = 'LossTP0MaxRatio'


def ModelFilePath():
    temp_path_name = "./model/dqn_fix/%s_%s_%u_%u_%u" % (
                        dqn_dataset.TrainSettingName(),
                        loss_func_, BATCH_SIZE, LSTM_SIZE, int(use_test_data))
    return temp_path_name

def ModelFileNames():
    temp_path_name = ModelFilePath()
    model_name = "%s/model.h5" % temp_path_name
    mean_name = "%s/mean.npy" % temp_path_name
    std_name = "%s/std.npy" % temp_path_name
    return temp_path_name, model_name, mean_name, std_name

def LossTanhDiff(y_true, y_pred, e=0.1):
    return abs(K.tanh((y_true - 5.0) * 0.4) - K.tanh((y_pred - 5.0) * 0.4))

def LossTP0MaxRatio(y_true, y_pred, e=0.1):
    return abs(y_true - y_pred) / 10.0 * K.max([y_true, y_pred, y_true*0.0])

def ActiveLoss():
    my_loss = ''
    if loss_func_ == 'LossTP0MaxRatio':
        my_loss = LossTP0MaxRatio
    elif loss_func_ == 'LossTanhDiff':
        my_loss = LossTanhDiff
    return my_loss

def BuildModel():
    # model
    model = keras.models.Sequential()
    model.add(keras.layers.LSTM(LSTM_SIZE, 
                                input_shape=(feature.feature_days, feature.feature_unit_size), 
                                return_sequences=False))
    model.add(keras.layers.Dense(1))

    my_optimizer = keras.optimizers.RMSprop(lr=0.004, rho=0.9, epsilon=1e-06)
    active_loss = ActiveLoss()
    model.compile(loss=active_loss, optimizer=my_optimizer, metrics=[active_loss])
    return model

def SaveModel(model, mean, std, epoch=-1):
    temp_path_name, model_name, mean_name, std_name = ModelFileNames()
    if not os.path.exists(temp_path_name):
        os.makedirs(temp_path_name)
    model.save(model_name)
    np.save(mean_name, mean)
    np.save(std_name, std)
    if epoch >= 0:
        model_name = '%s/model_%u.h5' % (temp_path_name, epoch)
        model.save(model_name)

def LoadModel():
    temp_path_name, model_name, mean_name, std_name = ModelFileNames()
    print("LoadModel: %s" % model_name)
    model = keras.models.load_model(model_name, custom_objects={loss_func_: ActiveLoss()})
    mean = np.load(mean_name)
    std = np.load(std_name)
    return model, mean, std

def ModelExist():
    # return False
    temp_path_name, model_name, mean_name, std_name = ModelFileNames()
    return (os.path.exists(model_name) and os.path.exists(mean_name) and os.path.exists(std_name))

def ReshapeRnnFeatures(features):
    return features.reshape(features.shape[0], feature.feature_days, feature.feature_unit_size)

def FeaturesPretreat(features, mean, std):
    features = (features - mean) / std
    features = ReshapeRnnFeatures(features)
    return features

def Plot2DArray(ax, arr, name, color=''):
    np_arr = np.array(arr)
    if len(np_arr) > 1:
        x = np_arr[:,0]
        y = np_arr[:,1]
        if color != '':
            ax.plot(x, y, color, label=name)
        else:
            ax.plot(x, y, label=name)

def PlotHistory(losses, val_losses, test_increase):
    if not hasattr(PlotHistory, 'fig'):
        plt.ion()
        PlotHistory.fig, PlotHistory.ax1 = plt.subplots()
        PlotHistory.ax2 = PlotHistory.ax1.twinx()
    PlotHistory.ax1.cla()
    PlotHistory.ax2.cla()
    Plot2DArray(PlotHistory.ax1, losses, 'loss')
    Plot2DArray(PlotHistory.ax1, val_losses, 'val_loss')
    Plot2DArray(PlotHistory.ax2, test_increase, 'test_increase', 'g-')
    PlotHistory.ax1.legend()
    # PlotHistory.ax2.legend()
    plt.show()
    # plt.pause(1)
    temp_path_name = ModelFilePath()
    if not os.path.exists(temp_path_name):
        os.makedirs(temp_path_name)
    plt.savefig('%s/figure.png' % temp_path_name)

def TestModel(input_test_data, input_model, input_mean, input_std):
    return 0.0

def Train():
    train_features, train_labels, val_features, val_labels = dqn_fix_dataset.GetDataSet()

    print("reorder...")
    order=np.argsort(np.random.random(len(train_labels)))
    train_features=train_features[order]
    train_labels=train_labels[order]

    print("pretreat...")
    where_are_nan = np.isnan(train_features)
    where_are_inf = np.isinf(train_features)
    train_features[where_are_nan] = 0.0
    train_features[where_are_inf] = 0.0
    mean = train_features.mean(axis=0)
    std = train_features.std(axis=0)
    # print("mean: {}".format(mean.shape))
    # print("std: {}".format(std.shape))
    train_features = FeaturesPretreat(train_features, mean, std)
    print("train_features: {}".format(train_features.shape))
    val_features = FeaturesPretreat(val_features, mean, std)

    model = BuildModel()
    model.summary()

    # Display training progress by printing a single dot for each completed epoch.
    class PrintDot(keras.callbacks.Callback):
        def on_epoch_end(self,epoch,logs):
            if epoch % 1 == 0: 
                sys.stdout.write('\r%d' % (epoch))
                sys.stdout.flush()
            #print('.', end='')
            #print('.')
    class TestCallback(keras.callbacks.Callback):
        def on_train_begin(self, logs={}):
            self.losses = []
            self.val_losses = []
            self.test_increase = []

        # def on_batch_end(self, batch, logs={}):
        #     self.losses.append(logs.get('loss'))

        def on_epoch_end(self, epoch, logs={}):
            sys.stdout.write('\r%d' % (epoch))
            sys.stdout.flush()
            if use_test_data:
                self.losses.append([epoch, logs.get('loss')])
                self.val_losses.append([epoch, logs.get('val_loss')])
                if ((epoch % 1) == 0):
                    self.test_increase.append([epoch, TestModel(test_data, self.model, mean, std)])
                SaveModel(self.model, mean, std, epoch)
                # SaveHistory(self.losses, self.val_losses, self.test_increase)
                PlotHistory(self.losses, self.val_losses, self.test_increase)
            else:
                self.losses.append([epoch, logs.get('loss')])
                self.val_losses.append([epoch, logs.get('val_loss')])
                # SaveHistory(self.losses, self.val_losses, self.test_increase)
                PlotHistory(self.losses, self.val_losses, self.test_increase)
                SaveModel(self.model, mean, std, epoch)



    # The patience parameter is the amount of epochs to check for improvement.
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)
    history = model.fit(train_features, train_labels, epochs=EPOCHS, batch_size=BATCH_SIZE, 
                        validation_data=(val_features, val_labels), verbose=0,
                        # callbacks=[early_stop, TestCallback()])
                        callbacks=[TestCallback()])


    # print("%-12s%-12s%-12s" %('epoch', 'train_err', 'val_err'))
    # for iloop in history.epoch:
    #     train_err=history.history['mean_absolute_error'][iloop]
    #     val_err=history.history['val_mean_absolute_error'][iloop]
    #     print("%8u%8.2f%8.2f" %(iloop, train_err, val_err))

    # # 显示 <<<<<<<<<<
    # import matplotlib.pyplot as plt
    # def plot_history(history):
    #     plt.figure()
    #     plt.xlabel('Epoch')
    #     plt.ylabel('loss')
    #     plt.plot(history.epoch, np.array(history.history['loss']), 
    #             label='Train Loss')
    #     plt.plot(history.epoch, np.array(history.history['val_loss']),
    #             label = 'Val loss')
    #     plt.legend()
    #     #plt.ylim([0,5])
    #     plt.show()
    # print("\nplot_history")
    # plot_history(history)
    # # 显示 >>>>>>>>>>>>

    SaveModel(model, mean, std)


def main(argv):
    del argv

    if FLAGS.test:
        model, mean, std = LoadModel()
        dqn_obj = dqn.DQN()
        dqn_obj.TestTop1LowLevel(model, mean, std, True)
    else:
        Train()

    exit()

if __name__ == "__main__":
    app.run(main)

    


