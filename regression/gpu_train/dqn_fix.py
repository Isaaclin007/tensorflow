# -*- coding:UTF-8 -*-


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
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import backend as K
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import dqn_fix_dataset
import loss
import dqn_test
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import_feature = os.path.exists('feature.py')
if import_feature:
    import feature

import_dqn = os.path.exists('dqn.py')
if import_dqn:
    import dqn

import_dqn_dataset = os.path.exists('dqn_dataset.py')
if import_dqn_dataset:
    import dqn_dataset

import_np_common = os.path.exists('../common/np_common.py')
if import_np_common:
    sys.path.append("..")
    from common import np_common

if import_dqn_dataset and import_feature:
    feature_feature_days_ = feature.feature_days
    feature_feature_unit_size_ = feature.feature_unit_size
else:
    feature_feature_days_ = 30
    feature_feature_unit_size_ = 5

FLAGS = flags.FLAGS


BATCH_SIZE = 10240
LSTM_SIZE = 8
LEARNING_RATE = 0.004
TRAIN_EPOCH = 500
use_test_data = False
loss_func_ = 'mean_absolute_tp0_max_ratio_error'
# loss_func_ = 'LossAbs'


def ModelFilePath():
    temp_path_name = "./model/dqn_fix/%s_%s_%u_%u_%f_%u" % (
                        dqn_fix_dataset.setting_name_,
                        loss_func_, BATCH_SIZE, LSTM_SIZE, LEARNING_RATE, int(use_test_data))
    return temp_path_name

def ModelFileNames(epoch=-1):
    temp_path_name = ModelFilePath()
    if epoch >= 0:
        model_name = '%s/model_%u.h5' % (temp_path_name, epoch)
    else:
        model_name = "%s/model.h5" % temp_path_name
    mean_name = "%s/mean.npy" % temp_path_name
    std_name = "%s/std.npy" % temp_path_name
    return temp_path_name, model_name, mean_name, std_name

def BuildModel():
    # model
    model = keras.models.Sequential()
    model.add(keras.layers.LSTM(LSTM_SIZE, 
                                input_shape=(feature_feature_days_, feature_feature_unit_size_), 
                                return_sequences=False))
    model.add(keras.layers.Dense(1))

    my_optimizer = keras.optimizers.RMSprop(lr=LEARNING_RATE, rho=0.9, epsilon=1e-06)
    active_loss = loss.LossFunc(loss_func_)
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

def LoadModel(epoch=-1):
    temp_path_name, model_name, mean_name, std_name = ModelFileNames(epoch)
    print("LoadModel: %s" % model_name)
    model = keras.models.load_model(model_name, custom_objects=loss.LossDict())
    mean = np.load(mean_name)
    std = np.load(std_name)
    return model, mean, std

def ModelExist():
    # return False
    temp_path_name, model_name, mean_name, std_name = ModelFileNames()
    return (os.path.exists(model_name) and os.path.exists(mean_name) and os.path.exists(std_name))

def ReshapeRnnFeatures(features):
    return features.reshape(features.shape[0], feature_feature_days_, feature_feature_unit_size_)

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

def PlotHistory(losses, val_losses, train_increase, test_increase):
    if len(losses) <= 1:
        return
    if not hasattr(PlotHistory, 'fig'):
        plt.ion()
        PlotHistory.fig, PlotHistory.ax1 = plt.subplots()
        PlotHistory.ax2 = PlotHistory.ax1.twinx()
    PlotHistory.ax1.cla()
    PlotHistory.ax2.cla()
    Plot2DArray(PlotHistory.ax1, losses, 'loss')
    Plot2DArray(PlotHistory.ax1, val_losses, 'val_loss')
    Plot2DArray(PlotHistory.ax2, train_increase, 'train_increase', 'r-')
    Plot2DArray(PlotHistory.ax2, test_increase, 'test_increase', 'g-')
    PlotHistory.ax1.legend()
    # PlotHistory.ax2.legend()
    # plt.show()
    # plt.pause(1)
    temp_path_name = ModelFilePath()
    if not os.path.exists(temp_path_name):
        os.makedirs(temp_path_name)
    plt.savefig('%s/figure.png' % temp_path_name)

def SaveHistory(losses, val_losses, test_increase):
    temp_path_name = ModelFilePath()
    if not os.path.exists(temp_path_name):
        os.makedirs(temp_path_name)
    if len(losses) > 0:
        np.save('%s/loss.npy' % temp_path_name, np.array(losses))
    if len(val_losses) > 0:
        np.save('%s/val_losses.npy' % temp_path_name, np.array(val_losses))
    if len(test_increase):
        np.save('%s/test_increase.npy' % temp_path_name, np.array(test_increase))
    if FLAGS.showloss:
        PlotHistory(losses, val_losses, [], test_increase)

def ShowHistory():
    temp_path_name = ModelFilePath()
    print(temp_path_name)
    if not os.path.exists(temp_path_name):
        print("ShowHistory.Error: path (%s) not exist" % temp_path_name)
        return
    
    losses = []
    val_losses = []
    test_increase = []
    train_increase = []

    temp_file_name = '%s/loss.npy' % temp_path_name
    if os.path.exists(temp_file_name):
        losses = np.load(temp_file_name).tolist()
    temp_file_name = '%s/val_losses.npy' % temp_path_name
    if os.path.exists(temp_file_name):
        val_losses = np.load(temp_file_name).tolist()
    temp_file_name = '%s/train_increase.npy' % temp_path_name
    if os.path.exists(temp_file_name):
        train_increase = np.load(temp_file_name).tolist()
    temp_file_name = '%s/test_increase.npy' % temp_path_name
    if os.path.exists(temp_file_name):
        test_increase = np.load(temp_file_name).tolist()
        # max_increase_epoch = 0.0
        # max_increase = 0.0
        # for iloop in range(0, len(test_increase)):
        #     print("%-8.0f: %.1f" % (test_increase[iloop][0], test_increase[iloop][1]))
        #     if max_increase < test_increase[iloop][1]:
        #         max_increase = test_increase[iloop][1]
        #         max_increase_epoch = test_increase[iloop][0]
        # print("-------------------------------")
        # print("max increase(%.0f): %.1f" % (max_increase_epoch, max_increase))
        # captions = ['epoch', 'increase']
        # data_df = pd.DataFrame(test_increase, columns=captions)
        # data_df.to_csv('%s/test_increase.csv' % temp_path_name)

    PlotHistory(losses, val_losses, train_increase, test_increase)

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
                if ((epoch % FLAGS.step) == 0):
                    self.test_increase.append([epoch, TestModel(test_data, self.model, mean, std)])
                    SaveModel(self.model, mean, std, epoch)
                    SaveHistory(self.losses, self.val_losses, self.test_increase)
            else:
                self.losses.append([epoch, logs.get('loss')])
                self.val_losses.append([epoch, logs.get('val_loss')])
                if ((epoch % FLAGS.step) == 0):
                    SaveModel(self.model, mean, std, epoch)
                    SaveHistory(self.losses, self.val_losses, self.test_increase)
                



    # The patience parameter is the amount of epochs to check for improvement.
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)
    if FLAGS.epoch > 0:
        TRAIN_EPOCH = FLAGS.epoch
    history = model.fit(train_features, train_labels, epochs=TRAIN_EPOCH, batch_size=BATCH_SIZE, 
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

def TestLossTP0MaxRatio(y_true, y_pred):
    return np.abs(y_true - y_pred) / 10.0 * np.maximum(np.maximum(y_true, y_pred), np.abs(y_true) * 0.0)

def TestLossTP0MaxRatioTF(y_true, y_pred):
    temp_loss = LossTP0MaxRatio(y_true, y_pred)
    with tf.Session() as sess:
        return temp_loss.eval()

def TestLossTP0MaxP1MaxRatioTF(y_true, y_pred):
    temp_loss = LossTP0MaxP1MaxRatio(y_true, y_pred)
    with tf.Session() as sess:
        return temp_loss.eval()

def TestLossTP0MaxP1MaxRatio(y_true, y_pred):
    return np.abs(y_true - y_pred) * np.maximum(np.maximum(y_true, y_pred), 0.0) * np.maximum(y_pred, 1.0) * 0.1
                
                


def TestLossAbs(y_true, y_pred):
    return np.abs(y_true - y_pred)

def LossTest(features, labels, model, mean, std, loss_func_list, caption_list):
    features_pretreat = FeaturesPretreat(features, mean, std)
    predict_result = model.predict(features_pretreat)
    for iloop in range(len(loss_func_list)):
        loss_func = loss_func_list[iloop]
        caption = caption_list[iloop]
        print('\n----------------------------- %s -----------------------------' % caption)
        print('labels:{}'.format(labels.shape))
        print('predict_result:{}'.format(predict_result.shape))
        loss = loss_func(labels, predict_result)
        print('global_loss[%u]:%f' % (len(loss), np.mean(loss)))

        filter_loss = loss[labels > 0]
        print('loss > 0[%u]:%f' % (len(filter_loss), np.mean(filter_loss)))

        filter_loss = loss[labels < 0]
        print('loss < 0[%u]:%f' % (len(filter_loss), np.mean(filter_loss)))

        filter_loss = loss[labels > 10]
        print('loss > 10[%u]:%f' % (len(filter_loss), np.mean(filter_loss)))

        print('\n\n')

def TestAll(dataset_option='test'):
    dqn_test_obj = dqn_test.DQNTest()
    dqn_test_obj.LoadDataset(dqn_fix_dataset.dqn_dataset_file_name_, 
                             dqn_fix_dataset.dqn_dataset_dataset_train_test_split_date_,
                             dataset_option)
    increase_list = []
    for iloop in range(0, 1000000, FLAGS.step):
        if dqn_test_obj.LoadModel(ModelFilePath(), iloop):
            increase_sum, trade_count, max_Q_mean = dqn_test_obj.TestTop1(False)
            increase_list.append([iloop, increase_sum])
            sys.stdout.write('\r%d' % (iloop))
            sys.stdout.flush()
        else:
            break
    if len(increase_list):
        np.save('%s/%s_increase.npy' % (ModelFilePath(), dataset_option), np.array(increase_list))

def main(argv):
    del argv

    if FLAGS.mode == 'train':
        Train()
        
    elif FLAGS.mode == 'test':
        dqn_test_obj = dqn_test.DQNTest()
        dqn_test_obj.LoadTestData(dqn_fix_dataset.dqn_dataset_file_name_, 
                                  dqn_fix_dataset.dqn_dataset_dataset_train_test_split_date_)
        dqn_test_obj.LoadModel(ModelFilePath(), FLAGS.epoch)
        dqn_test_obj.TestTop1(True)
    elif FLAGS.mode == 'testall':
        TestAll('train')
        TestAll('test')
    
    elif FLAGS.mode == 'show':
        ShowHistory()

    elif FLAGS.mode == 'loss':
        # y_true = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 100.0])
        # y_pred = np.array([-1.5, 0.5, 1.5, 2.5, 3.5, 100.5])
        # y_true = y_true.astype('float32')
        # y_pred = y_pred.astype('float32')
        # y_true = y_true.reshape((len(y_true), 1))
        # y_pred = y_pred.reshape((len(y_pred), 1))
        # print(y_true)
        # print(y_pred)
        # print('{}'.format(y_true.dtype))
        # print('{}'.format(y_pred.dtype))
        # temp_loss = LossTP0MaxP1MaxRatio(y_true, y_pred)
        # with tf.Session() as sess:
        #     print(temp_loss.eval())
        # exit()

        model, mean, std = LoadModel(FLAGS.epoch)
        train_features, train_labels, val_features, val_labels = dqn_fix_dataset.GetDataSet()
        # train_features = train_features.astype('float32')
        # train_labels = train_labels.astype('float32')
        # val_features = val_features.astype('float32')
        # val_labels = val_labels.astype('float32')
        # loss_func_list = [TestLossTP0MaxRatioTF, TestLossAbs]
        # caption_list = ['TP0MaxRatioTF', 'Abs']
        # LossTest(train_features, train_labels, model, mean, std, loss_func_list, caption_list)
        # caption_list = ['val.TP0MaxRatioTF', 'val.Abs']
        # LossTest(val_features, val_labels, model, mean, std, loss_func_list, caption_list)
        caption_list = ['TP0MaxP1MaxRatioTF', 'Abs']
        loss_func_list = [TestLossTP0MaxP1MaxRatioTF, TestLossAbs]
        LossTest(val_features, val_labels, model, mean, std, loss_func_list, caption_list)
    elif FLAGS.mode == 'map':
        train_features, train_labels, val_features, val_labels = dqn_fix_dataset.GetDataSet()
        map_label = (K.tanh((train_labels - 5.0) * 0.4) + 1.00001) * 5.0
        # map_label = K.tanh(train_labels * 0.4)
        with tf.Session() as sess:
            result = map_label.eval()
        # result = result[result < 0]
        print(result)
        np_common.ShowHist(result, 1)
        

    exit()

if __name__ == "__main__":
    flags.DEFINE_string('mode', 'train', 'train | test | testall | loss | show')
    flags.DEFINE_integer('epoch', -1, 'test model epoch')
    flags.DEFINE_integer('step', 1, 'train and testall model epoch step')
    flags.DEFINE_boolean('showloss', False, 'draw loss to model path figure.png')
    app.run(main)

    


