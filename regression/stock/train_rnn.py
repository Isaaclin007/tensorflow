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
import fix_dataset
import feature
import fix_test

HIDDEN_SIZE = 36
BATCH_SIZE = 10240
LEANING_RATE = 0.004
VAL_SPLIT = 0.1
train_mode = 'fix'
use_test_data = False

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
    # my_optimizer = keras.optimizers.RMSprop(lr=LEANING_RATE)
    model.compile(loss=mystockloss,
                    optimizer=my_optimizer,
                    metrics=[mystockloss])
    return model

def ModelFilePath(input_train_mode):
    if input_train_mode == "fix":
        temp_path_name = "./model/fix/%s_%u_%u_%f" % (fix_dataset.TrainSettingName(), HIDDEN_SIZE, BATCH_SIZE, LEANING_RATE)
    else:
        temp_path_name = "./model/wave/%s_%u_%u_%f" % (fix_dataset.SettingName(), HIDDEN_SIZE, BATCH_SIZE, LEANING_RATE)
    return temp_path_name

def ModelFileNames(input_train_mode, epoch=-1):
    temp_path_name = ModelFilePath(input_train_mode)
    if epoch == -1:
        model_name = "%s/model.h5" % temp_path_name
    else:
        model_name = "%s/model_%u.h5" % (temp_path_name, epoch)
    mean_name = "%s/mean.npy" % temp_path_name
    std_name = "%s/std.npy" % temp_path_name
    return temp_path_name, model_name, mean_name, std_name

def SaveModel(model, mean, std, epoch=-1):
    temp_path_name, model_name, mean_name, std_name = ModelFileNames(train_mode, epoch)
    if not os.path.exists(temp_path_name):
        os.makedirs(temp_path_name)
    model.save(model_name)
    if (epoch == -1) or (epoch == 0):
        np.save(mean_name, mean)
        np.save(std_name, std)

def LoadModel(input_train_mode, epoch=-1):
    temp_path_name, model_name, mean_name, std_name = ModelFileNames(input_train_mode, epoch)
    model = keras.models.load_model(model_name)
    mean = np.load(mean_name)
    std = np.load(std_name)
    return model, mean, std

def ReshapeRnnFeatures(features):
    return features.reshape(features.shape[0], feature.feature_days, feature.feature_unit_size)

def FeaturesPretreat(features, mean, std):
    features = (features - mean) / std
    features = ReshapeRnnFeatures(features)
    return features

def GetTestData():
    if train_mode == 'fix':
        return fix_dataset.GetTestData()

def TestModel(input_test_data, input_model, input_mean, input_std):
    # my_optimizer = tf.train.RMSPropOptimizer(LEANING_RATE)
    # my_optimizer = tf.keras.optimizers.RMSProp(LEANING_RATE)
    # input_model.compile(loss=mystockloss,
    #                 optimizer=my_optimizer,
    #                 metrics=[mystockloss])
    if train_mode == 'fix':
        return fix_test.TestEntry(input_test_data, False, input_model, input_mean, input_std)

import matplotlib.pyplot as plt
def PlotHistory(losses, val_losses, test_increase):
    data_num = len(losses)
    if data_num > 0:
        plt.ion()
        plt.cla()
        x = range(0, data_num)
        plt.plot(x, np.array(losses), label='train Loss')
        plt.plot(x, np.array(val_losses), label='val Loss')
        if len(test_increase) > 0:
            plt.plot(x, np.array(test_increase), label='test increase')
        plt.legend()
        plt.show()
        plt.pause(0.001)
        temp_path_name = ModelFilePath(train_mode)
        if not os.path.exists(temp_path_name):
            os.makedirs(temp_path_name)
        plt.savefig('%s/figure.png' % temp_path_name)

def train():
    if train_mode == "fix":
        train_features, train_labels = fix_dataset.GetTrainData()
    else:
        train_features, train_labels = wave_kernel.GetTrainData()
    # train_features = tushare_data.Features10D14To10D5(train_features)
    print("train_features: {}".format(train_features.shape))

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

    if use_test_data:
        test_data = GetTestData()
        val_features = test_data[:, 0:feature.FEATURE_SIZE()]
        val_features = FeaturesPretreat(val_features, mean, std)
        val_labels = test_data[:, feature.FEATURE_SIZE():feature.FEATURE_SIZE()+1]
        print("val_features: {}".format(val_features.shape))
    else:
        print("split...")
        val_data_num = int(len(train_labels) * VAL_SPLIT)
        val_features = train_features[:val_data_num]
        val_labels = train_labels[:val_data_num]
        train_features = train_features[val_data_num:]
        train_labels = train_labels[val_data_num:]
        print("train_features: {}".format(train_features.shape))
        print("val_features: {}".format(val_features.shape))

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
                if ((epoch % 5) == 0):
                    self.losses.append(logs.get('loss'))
                    self.val_losses.append(logs.get('val_loss'))
                    self.test_increase.append(TestModel(test_data, self.model, mean, std) / 10.0)
                    PlotHistory(self.losses, self.val_losses, self.test_increase)
                    SaveModel(self.model, mean, std, epoch)
            else:
                self.losses.append(logs.get('loss'))
                self.val_losses.append(logs.get('val_loss'))
                PlotHistory(self.losses, self.val_losses, self.test_increase)
                if ((epoch % 5) == 0):
                    SaveModel(self.model, mean, std, epoch)



    # The patience parameter is the amount of epochs to check for improvement.
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)
    history = model.fit(train_features, train_labels, epochs=EPOCHS, batch_size=BATCH_SIZE, 
                        validation_data=(val_features, val_labels), verbose=0,
                        callbacks=[early_stop, TestCallback()])
                        # callbacks=[PrintDot()])


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

    # SaveModel(model, mean, std)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        train_mode = sys.argv[1]
    train()

