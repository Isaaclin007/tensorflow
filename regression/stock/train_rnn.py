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
import matplotlib.pyplot as plt
import wave_dataset
import wave_kernel
from tensorflow.python.keras.callbacks import LearningRateScheduler
import fix_dataset
import feature
import fix_test
import wave_test_regression as wave_test


BATCH_SIZE = 10240
LEANING_RATE = 0.004
VAL_SPLIT = 0.2
train_mode = 'fix'
use_test_data = True

# MODEL_DENSE_4_TP0MaxRatio = 'd4_TP0MaxRatio'
# MODEL_LSTM_4 = '4'
# MODEL_LSTM_16 = '16'
# MODEL_LSTM_32 = '32'
# MODEL_LSTM_36 = '36'
# MODEL_LSTM_4_TP0MaxRatio = 'LSTM4_TP0MaxRatio'
# MODEL_LSTM_36_TP0MaxRatio = 'LSTM36_TP0MaxRatio'
# MODEL_LSTM_36_TP0MaxRatio_D4 = 'LSTM36_TP0MaxRatio_D4'
# MODEL_LSTM_36_TP10MaxRatio = 'LSTM36_TP10MaxRatio'

model_type = 'LSTM'
lstm_size = 36
lstm_dense_size = 1
optimizer_type = 'RMSProp'
loss_func = 'TP0MaxRatio'
model_option = '%s_%u.%u_%s_%s' % (model_type, lstm_size, lstm_dense_size, optimizer_type, loss_func)
reshape_data_rnn = True

SPLIT_MODE_SAMPLE_BY_DATE = 'samplebydate'
SPLIT_MODE_SPLIT_BY_DATE = 'splitbydate'
SPLIT_MODE_RANDOM = 'random'
train_test_split_mode = SPLIT_MODE_SPLIT_BY_DATE

reload(sys)
sys.setdefaultencoding('utf-8')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

def LossTP0MaxRatio(y_true, y_pred, e=0.1):
    return abs(y_true - y_pred) / 10.0 * K.max([y_true, y_pred, y_true*0.0])

def LossT10P0MaxRatio(y_true, y_pred, e=0.1):
    return abs(y_true - y_pred) / 10.0 * K.max([y_true, y_pred * 10.0, y_true*0.0])

def LossT2P0MaxRatio(y_true, y_pred, e=0.1):
    return abs(y_true - y_pred) / 10.0 * K.max([y_true, y_pred * 2.0, y_true*0.0])

def LossTP1MaxRatio(y_true, y_pred, e=0.1):
    return abs(y_true - y_pred) / 10.0 * K.max([y_true, y_pred, (abs(y_true) + 100) / (abs(y_true) + 100)])

def LossTs5Ps50MaxRatio(y_true, y_pred, e=0.1):
    return abs(y_true - y_pred) / 10.0 * K.max([(y_true - 5.0), (y_pred - 5.0), y_true*0.0])

def mystockloss(y_true, y_pred, e=0.1):
    return abs(y_true - y_pred)

def LossTanhDiff(y_true, y_pred, e=0.1):
    return abs(K.tanh((y_true - 5.0) * 0.4) - K.tanh((y_pred - 5.0) * 0.4))

def build_model(input_layer_shape):
    # model
    if model_type == 'LSTM':
        model = keras.models.Sequential()
        model.add(keras.layers.LSTM(lstm_size, input_shape=(input_layer_shape), return_sequences=False))
        model.add(keras.layers.Dense(lstm_dense_size))
    # optimizer
    if optimizer_type == 'RMSProp':
        my_optimizer = tf.train.RMSPropOptimizer(LEANING_RATE)
    
    # loss
    if loss_func == 'TP0MaxRatio':
        my_loss = LossTP0MaxRatio

    model.compile(loss=LossTP0MaxRatio, optimizer=my_optimizer, metrics=[LossTP0MaxRatio])
    return model
    # if model_option == MODEL_DENSE_4_TP0MaxRatio:
    #     model = keras.Sequential([
    #         keras.layers.Dense(4, activation=tf.nn.relu, input_shape=input_layer_shape),
    #         # keras.layers.Dense(128, activation=tf.nn.relu),
    #         keras.layers.Dense(1)
    #     ])
    #     optimizer = tf.train.RMSPropOptimizer(LEANING_RATE)
    #     model.compile(loss=LossTP0MaxRatio, optimizer=optimizer, metrics=[LossTP0MaxRatio])
    # elif model_option == MODEL_LSTM_4:
    #     model = keras.models.Sequential()
    #     # model.add(keras.layers.Dense(HIDDEN_SIZE, input_shape=(input_layer_shape)))
    #     # model.add(keras.layers.SimpleRNN(HIDDEN_SIZE, return_sequences=False, input_shape=(input_layer_shape),unroll=True))
    #     model.add(keras.layers.LSTM(4, input_shape=(input_layer_shape), return_sequences=False))
    #     # model.add(keras.layers.LSTM(32, input_shape=(input_layer_shape), return_sequences=True))
    #     # model.add(keras.layers.LSTM(32, return_sequences=True))
    #     # model.add(keras.layers.LSTM(10))
    #     model.add(keras.layers.Dense(1))
    #     # model.compile(loss="mae", optimizer="rmsprop")
    #     my_optimizer = tf.train.RMSPropOptimizer(LEANING_RATE)
    #     # my_optimizer = keras.optimizers.RMSprop(lr=LEANING_RATE)
    #     model.compile(loss=mystockloss, optimizer=my_optimizer, metrics=[mystockloss])
    # elif model_option == MODEL_LSTM_16:
    #     model = keras.models.Sequential()
    #     model.add(keras.layers.LSTM(16, input_shape=(input_layer_shape), return_sequences=False))
    #     model.add(keras.layers.Dense(1))
    #     my_optimizer = tf.train.RMSPropOptimizer(LEANING_RATE)
    #     model.compile(loss=mystockloss, optimizer=my_optimizer, metrics=[mystockloss])
    # elif model_option == MODEL_LSTM_32:
    #     model = keras.models.Sequential()
    #     model.add(keras.layers.LSTM(32, input_shape=(input_layer_shape), return_sequences=False))
    #     model.add(keras.layers.Dense(1))
    #     my_optimizer = tf.train.RMSPropOptimizer(LEANING_RATE)
    #     model.compile(loss=mystockloss, optimizer=my_optimizer, metrics=[mystockloss])
    # elif model_option == MODEL_LSTM_36:
    #     model = keras.models.Sequential()
    #     model.add(keras.layers.LSTM(36, input_shape=(input_layer_shape), return_sequences=False))
    #     model.add(keras.layers.Dense(1))
    #     my_optimizer = tf.train.RMSPropOptimizer(LEANING_RATE)
    #     model.compile(loss=mystockloss, optimizer=my_optimizer, metrics=[mystockloss])
    # elif model_option == MODEL_LSTM_36_TP0MaxRatio:
    #     model = keras.models.Sequential()
    #     model.add(keras.layers.LSTM(36, input_shape=(input_layer_shape), return_sequences=False))
    #     model.add(keras.layers.Dense(1))
    #     my_optimizer = tf.train.RMSPropOptimizer(LEANING_RATE)
    #     model.compile(loss=LossTP0MaxRatio, optimizer=my_optimizer, metrics=[LossTP0MaxRatio])
    # elif model_option == MODEL_LSTM_36_TP0MaxRatio_D4:
    #     model = keras.models.Sequential()
    #     model.add(keras.layers.LSTM(36, input_shape=(input_layer_shape), return_sequences=False))
    #     model.add(keras.layers.Dense(4))
    #     model.add(keras.layers.Dense(1))
    #     my_optimizer = tf.train.RMSPropOptimizer(LEANING_RATE)
    #     model.compile(loss=LossTP0MaxRatio, optimizer=my_optimizer, metrics=[LossTP0MaxRatio])
    # elif model_option == MODEL_LSTM_36_TP10MaxRatio:
    #     model = keras.models.Sequential()
    #     model.add(keras.layers.LSTM(36, input_shape=(input_layer_shape), return_sequences=False))
    #     model.add(keras.layers.Dense(1))
    #     my_optimizer = tf.train.RMSPropOptimizer(LEANING_RATE)
    #     model.compile(loss=LossTP10MaxRatio, optimizer=my_optimizer, metrics=[LossTP10MaxRatio])
    # return model

def ModelFilePath(input_train_mode):
    if input_train_mode == "fix":
        temp_path_name = "./model/fix/%s_%s_%u_%f_%s" % (
                         fix_dataset.TrainSettingName(), 
                         model_option, 
                         BATCH_SIZE, 
                         LEANING_RATE, 
                         train_test_split_mode)
    else:
        temp_path_name = "./model/wave/%s_%s_%s_%u_%f" % (
                         fix_dataset.SettingName(), 
                         wave_kernel.SettingName(),
                         model_option, 
                         BATCH_SIZE, 
                         LEANING_RATE)
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

def LoadTestModel(input_train_mode):
    if input_train_mode == 'fix':
        model_name = "./model/fix/model.h5"
        mean_name = "./model/fix/mean.npy"
        std_name = "./model/fix/std.npy"
    elif input_train_mode == 'wave':
        model_name = "./model/wave/model.h5"
        mean_name = "./model/wave/mean.npy"
        std_name = "./model/wave/std.npy"
    model = keras.models.load_model(model_name, custom_objects={
        'LossTP0MaxRatio': LossTP0MaxRatio,
        'LossTP1MaxRatio': LossTP1MaxRatio, 
        'LossT10P0MaxRatio': LossT10P0MaxRatio,
        'LossT2P0MaxRatio': LossT2P0MaxRatio,
        'LossTs5Ps50MaxRatio': LossTs5Ps50MaxRatio, 
        'LossTanhDiff': LossTanhDiff})
    mean = np.load(mean_name)
    std = np.load(std_name)
    return model, mean, std

def ReshapeRnnFeatures(features):
    return features.reshape(features.shape[0], feature.feature_days, feature.feature_unit_size)

def FeaturesPretreat(features, mean, std):
    features = (features - mean) / std
    if reshape_data_rnn:
        features = ReshapeRnnFeatures(features)
    return features

def TestModel(input_test_data, input_model, input_mean, input_std):
    # my_optimizer = tf.train.RMSPropOptimizer(LEANING_RATE)
    # my_optimizer = tf.keras.optimizers.RMSProp(LEANING_RATE)
    # input_model.compile(loss=mystockloss,
    #                 optimizer=my_optimizer,
    #                 metrics=[mystockloss])
    if train_mode == 'fix':
        return fix_test.TestEntry(input_test_data, False, input_model, input_mean, input_std)
    else:
        return wave_test.TestEntry(input_test_data, False, input_model, input_mean, input_std)

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
    temp_path_name = ModelFilePath(train_mode)
    if not os.path.exists(temp_path_name):
        os.makedirs(temp_path_name)
    plt.savefig('%s/figure.png' % temp_path_name)
    # data_num = len(losses)
    # if data_num > 0:
    #     plt.ion()
    #     plt.cla()
    #     x = range(0, data_num)
    #     plt.plot(x, np.array(losses), label='train Loss')
    #     plt.plot(x, np.array(val_losses), label='val Loss')
    #     if len(test_increase) > 0:
    #         plt.plot(x, np.array(test_increase), label='test increase')
    #     plt.legend()
    #     plt.show()
    #     plt.pause(0.001)
    #     temp_path_name = ModelFilePath(train_mode)
    #     if not os.path.exists(temp_path_name):
    #         os.makedirs(temp_path_name)
    #     plt.savefig('%s/figure.png' % temp_path_name)

def train():
    if train_mode == "fix":
        # train_features, train_labels = fix_dataset.GetTrainData()
        # train_features, train_labels, val_features, val_labels, test_data = fix_dataset.GetTrainTestData()
        if train_test_split_mode == SPLIT_MODE_SAMPLE_BY_DATE:
            train_features, train_labels, val_features, val_labels, test_data = fix_dataset.GetTrainTestDataSampleByDate(VAL_SPLIT)
        if train_test_split_mode == SPLIT_MODE_SPLIT_BY_DATE:
            train_features, train_labels, val_features, val_labels, test_data = fix_dataset.GetTrainTestDataSplitByDate()
        elif train_test_split_mode == SPLIT_MODE_RANDOM:
            train_features, train_labels, val_features, val_labels, test_data = fix_dataset.GetTrainTestDataRandom(VAL_SPLIT)
    else:
        if train_test_split_mode == SPLIT_MODE_SAMPLE_BY_DATE:
            train_features, train_labels, val_features, val_labels, test_data = wave_dataset.GetTrainTestDataSampleByDate(VAL_SPLIT)
        if train_test_split_mode == SPLIT_MODE_SPLIT_BY_DATE:
            train_features, train_labels, val_features, val_labels, test_data = wave_dataset.GetTrainTestDataSplitByDate()
        elif train_test_split_mode == SPLIT_MODE_RANDOM:
            train_features, train_labels, val_features, val_labels, test_data = wave_dataset.GetTrainTestDataRandom(VAL_SPLIT)
        # train_features, train_labels = wave_kernel.GetTrainData()
    # train_features = tushare_data.Features10D14To10D5(train_features)
    print("train_features: {}".format(train_features.shape))

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

    # if use_test_data:
    #     test_data = GetTestData()
    #     val_features = test_data[:, 0:feature.FEATURE_SIZE()]
    #     val_features = FeaturesPretreat(val_features, mean, std)
    #     val_labels = test_data[:, feature.FEATURE_SIZE():feature.FEATURE_SIZE()+1]
    #     print("val_features: {}".format(val_features.shape))
    # else:
    #     print("split...")
    #     val_data_num = int(len(train_labels) * VAL_SPLIT)
    #     val_features = train_features[:val_data_num]
    #     val_labels = train_labels[:val_data_num]
    #     train_features = train_features[val_data_num:]
    #     train_labels = train_labels[val_data_num:]
    #     print("train_features: {}".format(train_features.shape))
    #     print("val_features: {}".format(val_features.shape))

    # max_label_value = tushare_data.predict_day_count * 10.0
    # temp_mask = train_labels >= max_label_value
    # train_labels[temp_mask] = max_label_value

    model = build_model(train_features.shape[1:])
    model.summary()

    EPOCHS = 200

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

if __name__ == "__main__":
    if len(sys.argv) > 1:
        train_mode = sys.argv[1]
    if len(sys.argv) > 2 and sys.argv[2] == 'show':
        ShowHistory()
    else:
        train()

