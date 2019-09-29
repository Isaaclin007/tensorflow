# -*- coding:UTF-8 -*-

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import backend as K
import numpy as np
import pandas as pd
import os
import time
import sys
import math
import getopt
import matplotlib.pyplot as plt
from tensorflow.python.keras.callbacks import LearningRateScheduler
import gpu_train_fix_dataset as fix_dataset
import gpu_train_wave_dataset as wave_dataset
import gpu_train_fix_test as fix_test
import gpu_train_wave_test as wave_test
import gpu_train_feature as feature

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

reload(sys)
sys.setdefaultencoding('utf-8')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

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

def LossTs5Ps50MaxRatioMean(y_true, y_pred, e=0.1):
    return K.mean(K.abs(y_true - y_pred) / 10.0 * K.max([(y_true - 5.0), (y_pred - 5.0), y_true * 0.0]))

def LossTs9Ps90MaxRatio(y_true, y_pred, e=0.1):
    return abs(y_true - y_pred) / 10.0 * K.max([(y_true - 9.0), (y_pred - 9.0), y_true*0.0])

def LossAbs(y_true, y_pred, e=0.1):
    return abs(y_true - y_pred)

def LossTP010ClipDiff(y_true, y_pred, e=0.1):
    return K.mean(K.abs(K.clip(y_true, 0, 10) - K.clip(y_pred, 0, 10)))

def LossTanhDiff(y_true, y_pred, e=0.1):
    return abs(K.tanh((y_true - 5.0) * 0.4) - K.tanh((y_pred - 5.0) * 0.4))


# loss
def ActiveLoss():
    my_loss = ''
    if loss_func_ == 'LossTP0MaxRatio':
        my_loss = LossTP0MaxRatio
    elif loss_func_ == 'LossTP1MaxRatio':
        my_loss = LossTP1MaxRatio
    elif loss_func_ == 'LossT10P0MaxRatio':
        my_loss = LossT10P0MaxRatio
    elif loss_func_ == 'LossT2P0MaxRatio':
        my_loss = LossT2P0MaxRatio
    elif loss_func_ == 'LossTs5Ps50MaxRatio':
        my_loss = LossTs5Ps50MaxRatio
    elif loss_func_ == 'LossTs5Ps50MaxRatioMean':
        my_loss = LossTs5Ps50MaxRatioMean
    elif loss_func_ == 'LossTs9Ps90MaxRatio':
        my_loss = LossTs9Ps90MaxRatio
    elif loss_func_ == 'LossAbs':
        my_loss = LossAbs
    elif loss_func_ == 'LossTP010ClipDiff':
        my_loss = LossTP010ClipDiff
    elif loss_func_ == 'LossTanhDiff':
        my_loss = LossTanhDiff
    return my_loss

# optimizer
def ActiveOptimizer():
    if optimizer_ == 'RMSProp':
        my_optimizer = tf.train.RMSPropOptimizer(learning_rate_)
    elif optimizer_ == 'KerasRMSProp':
        my_optimizer = keras.optimizers.RMSprop(lr=learning_rate_, rho=0.9, epsilon=1e-06)
    return my_optimizer

def build_model(input_layer_shape):
    # model
    if model_type_ == 'LSTM':
        model = keras.models.Sequential()
        model.add(keras.layers.LSTM(lstm_size_, input_shape=(input_layer_shape), return_sequences=False))
        model.add(keras.layers.Dense(lstm_dense_size_))
    elif model_type_ == 'Dense':
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(dense_size_[0], activation=tf.nn.relu, input_shape=input_layer_shape))
        for iloop in range(1, dense_layer_num_):
            model.add(keras.layers.Dense(dense_size_[iloop], activation=tf.nn.relu))
        model.add(keras.layers.Dense(1))

    if loss_func_ == 'mae':
        model.compile(loss="mae", optimizer=ActiveOptimizer())
    else:
        model.compile(loss=ActiveLoss(), optimizer=ActiveOptimizer(), metrics=[ActiveLoss()])
    return model

def ModelFilePath(input_train_mode):
    temp_path_name = "./model/%s/%s_%s" % (train_data_, SettingName(), feature.SettingName())
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
    temp_path_name, model_name, mean_name, std_name = ModelFileNames(train_data_, epoch)
    if not os.path.exists(temp_path_name):
        os.makedirs(temp_path_name)
    model.save(model_name)
    if (epoch == -1) or (epoch == 0):
        np.save(mean_name, mean)
        np.save(std_name, std)

def LoadModel(input_train_mode, epoch=-1):
    temp_path_name, model_name, mean_name, std_name = ModelFileNames(input_train_mode, epoch)
    print("LoadModel: %s" % model_name)
    model = keras.models.load_model(model_name, custom_objects={loss_func_: ActiveLoss()})
    mean = np.load(mean_name)
    std = np.load(std_name)
    return model, mean, std

def ReshapeRnnFeatures(features):
    return features.reshape(features.shape[0], feature.feature_days, feature.feature_unit_size)

def FeaturesPretreat(features, mean, std):
    features = (features - mean) / std
    if model_type_ == 'LSTM':
        features = ReshapeRnnFeatures(features)
    return features

def TestModel(input_test_data, input_model, input_mean, input_std):
    if train_data_ == 'fix':
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
    plt.show()
    plt.pause(5)
    temp_path_name = ModelFilePath(train_data_)
    if not os.path.exists(temp_path_name):
        os.makedirs(temp_path_name)
    plt.savefig('%s/figure.png' % temp_path_name)

def SaveHistory(losses, val_losses, test_increase):
    temp_path_name = ModelFilePath(train_data_)
    if not os.path.exists(temp_path_name):
        os.makedirs(temp_path_name)
    if len(losses) > 0:
        np.save('%s/loss.npy' % temp_path_name, np.array(losses))
    if len(val_losses) > 0:
        np.save('%s/val_losses.npy' % temp_path_name, np.array(val_losses))
    if len(test_increase):
        np.save('%s/test_increase.npy' % temp_path_name, np.array(test_increase))

def ShowHistory():
    temp_path_name = ModelFilePath(train_data_)
    if not os.path.exists(temp_path_name):
        print("ShowHistory.Error: path (%s) not exist" % temp_path_name)
        return
    
    losses = []
    val_losses = []
    test_increase = []

    temp_file_name = '%s/loss.npy' % temp_path_name
    if os.path.exists(temp_file_name):
        losses = np.load(temp_file_name).tolist()
    temp_file_name = '%s/val_losses.npy' % temp_path_name
    if os.path.exists(temp_file_name):
        val_losses = np.load(temp_file_name).tolist()
    temp_file_name = '%s/test_increase.npy' % temp_path_name
    if os.path.exists(temp_file_name):
        test_increase = np.load(temp_file_name).tolist()
        max_increase_epoch = 0.0
        max_increase = 0.0
        for iloop in range(0, len(test_increase)):
            print("%-8.0f: %.1f" % (test_increase[iloop][0], test_increase[iloop][1]))
            if max_increase < test_increase[iloop][1]:
                max_increase = test_increase[iloop][1]
                max_increase_epoch = test_increase[iloop][0]
        print("-------------------------------")
        print("max increase(%.0f): %.1f" % (max_increase_epoch, max_increase))
        captions = ['epoch', 'increase']
        data_df = pd.DataFrame(test_increase, columns=captions)
        data_df.to_csv('%s/test_increase.csv' % temp_path_name)

    PlotHistory(losses, val_losses, test_increase)

def train():
    if train_data_ == "fix":
        if data_split_mode_ == 'samplebydate':
            train_features, train_labels, val_features, val_labels, test_data = fix_dataset.GetTrainTestDataSampleByDate(val_split_)
        if data_split_mode_ == 'splitbydate':
            train_features, train_labels, val_features, val_labels, test_data = fix_dataset.GetTrainTestDataSplitByDate()
        elif data_split_mode_ == 'random':
            train_features, train_labels, val_features, val_labels, test_data = fix_dataset.GetTrainTestDataRandom(val_split_)
    else:
        if data_split_mode_ == 'samplebydate':
            train_features, train_labels, val_features, val_labels, test_data = wave_dataset.GetTrainTestDataSampleByDate(val_split_)
        if data_split_mode_ == 'splitbydate':
            train_features, train_labels, val_features, val_labels, test_data = wave_dataset.GetTrainTestDataSplitByDate()
        elif data_split_mode_ == 'random':
            train_features, train_labels, val_features, val_labels, test_data = wave_dataset.GetTrainTestDataRandom(val_split_)
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
    train_features = FeaturesPretreat(train_features, mean, std)
    print("train_features: {}".format(train_features.shape))

    val_features = FeaturesPretreat(val_features, mean, std)

    model = build_model(train_features.shape[1:])
    model.summary()

    # Display training progress by printing a single dot for each completed epoch.
    class PrintDot(keras.callbacks.Callback):
        def on_epoch_end(self,epoch,logs):
            if epoch % 1 == 0: 
                sys.stdout.write('\r%d' % (epoch))
                sys.stdout.flush()
    class TrainCallback(keras.callbacks.Callback):
        def on_train_begin(self, logs={}):
            self.losses = []
            self.val_losses = []
            self.test_increase = []

        def on_epoch_end(self, epoch, logs={}):
            sys.stdout.write('\r%d' % (epoch))
            sys.stdout.flush()
            if use_test_data_:
                self.losses.append([epoch, logs.get('loss')])
                self.val_losses.append([epoch, logs.get('val_loss')])
                if ((epoch % 5) == 0):
                    self.test_increase.append([epoch, TestModel(test_data, self.model, mean, std)])
                    SaveModel(self.model, mean, std, epoch)
                SaveHistory(self.losses, self.val_losses, self.test_increase)
                # PlotHistory(self.losses, self.val_losses, self.test_increase)
            else:
                self.losses.append([epoch, logs.get('loss')])
                self.val_losses.append([epoch, logs.get('val_loss')])
                SaveHistory(self.losses, self.val_losses, self.test_increase)
                # PlotHistory(self.losses, self.val_losses, self.test_increase)
                if ((epoch % 5) == 0):
                    SaveModel(self.model, mean, std, epoch)



    # The patience parameter is the amount of epochs to check for improvement.
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)
    history = model.fit(train_features, train_labels, epochs=epochs_, batch_size=batch_size_, 
                        validation_data=(val_features, val_labels), verbose=0,
                        # callbacks=[early_stop, TestCallback()])
                        callbacks=[TrainCallback()])

    SaveModel(model, mean, std)

def InitParas(argv):
    opts,args = getopt.getopt(argv[1:],'-h-v', ['help',
                                                'version',
                                                'train_data=',
                                                'data_split_mode=',
                                                'val_split=',
                                                'model_type=',
                                                'lstm_size=',
                                                'lstm_dense_size=',
                                                'dense_layer_num=',
                                                'dense_size=',
                                                'loss_func=',
                                                'optimizer=',
                                                'fix_activer_label_day=',
                                                'learning_rate=',
                                                'batch_size=',
                                                'epoch='
                                                ])
    if len(args) > 0:
        print('InitParas.Error, unsupported args:')
        print(args)
        return False
    for opt_name,opt_value in opts:
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
        elif opt_name == '--dense_layer_num':
            dense_layer_num_ = int(opt_value)
        elif opt_name == '--dense_size':
            dense_size_ = map(int, opt_value.split(','))
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
        else:
            print('InitParas.Error, unsupported opt_name(%s)!' % opt_name)
            return False
    return True


if __name__ == "__main__":
    if InitParas(sys.argv):
        train()

