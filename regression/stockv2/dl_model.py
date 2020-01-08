# -*- coding:UTF-8 -*-


import numpy as np
import pandas as pd
import os
import time
import datetime
import sys
import math
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import backend as K
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import loss
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def Plot2DArray(ax, arr, name, color=''):
    np_arr = np.array(arr)
    if len(np_arr) > 1:
        x = np_arr[:,0]
        y = np_arr[:,1]
        if color != '':
            ax.plot(x, y, color, label=name)
        else:
            ax.plot(x, y, label=name)

def PlotHistory(save_path, losses, val_losses, train_increase, test_increase):
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
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig('%s/figure.png' % save_path)

class DLModel():
    def __init__(self, 
                 app_setting_name,
                 feature_unit_num,
                 feature_unit_size,
                 lstm_size,
                 batch_size,
                 learning_rate,
                 loss,
                 save_step=1,
                 test_func=None,
                 test_param=None):
        self.setting_name = '%s_%u_%u_%u_%u_%f_%s' % (app_setting_name, 
                                                      feature_unit_num, 
                                                      feature_unit_size, 
                                                      lstm_size, 
                                                      batch_size, 
                                                      learning_rate, 
                                                      loss)
        self.feature_unit_num = feature_unit_num
        self.feature_unit_size = feature_unit_size
        self.lstm_size = lstm_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.loss = loss
        self.model_path = './model/%s' % self.setting_name
        self.save_step = save_step
        self.test_func = test_func
        self.test_param = test_param
        self.continue_train = True
        self.losses = []
        self.val_losses = []
        self.test_increase = []
        self.init_epoch = 0
        
# BATCH_SIZE = 10240
# LSTM_SIZE = 8
# LEARNING_RATE = 0.004
# TRAIN_EPOCH = 500
# use_test_data = False
# loss_func_ = 'mean_absolute_tp0_max_ratio_error'
# # loss_func_ = 'LossAbs'

    def ModelFileNames(self, epoch=-1):
        temp_path_name = self.model_path
        if epoch >= 0:
            model_name = '%s/model_%u.h5' % (temp_path_name, epoch)
        else:
            model_name = "%s/model.h5" % temp_path_name
        mean_name = "%s/mean.npy" % temp_path_name
        std_name = "%s/std.npy" % temp_path_name
        return temp_path_name, model_name, mean_name, std_name

    def BuildModel(self):
        # model
        model = keras.models.Sequential()
        model.add(keras.layers.LSTM(self.lstm_size, 
                                    input_shape=(self.feature_unit_num, self.feature_unit_size), 
                                    return_sequences=False))
        model.add(keras.layers.Dense(1))

        my_optimizer = keras.optimizers.RMSprop(lr=self.learning_rate, rho=0.9, epsilon=1e-06)
        active_loss = loss.LossFunc(self.loss)
        model.compile(loss=active_loss, optimizer=my_optimizer, metrics=[active_loss])
        return model

    def SaveModel(self, model, mean, std, epoch=-1):
        temp_path_name, model_name, mean_name, std_name = self.ModelFileNames(epoch)
        if not os.path.exists(temp_path_name):
            os.makedirs(temp_path_name)
        model.save(model_name)
        np.save(mean_name, mean)
        np.save(std_name, std)

    def LoadHistoryUnit(self, his_name):
        file_name = '%s/%s.npy' % (self.model_path, his_name)
        if os.path.exists(file_name):
            return np.load(file_name).tolist()
        else:
            return []

    def MaxModelEpoch(self):
        losses = self.LoadHistoryUnit('loss')
        if len(losses) == 0:
            return 0
        else:
            return losses[-1][0]

    def LoadModel(self, epoch=-1):
        if epoch == -1:
            load_epoch = self.MaxModelEpoch()
        else:
            load_epoch = epoch
        temp_path_name, model_name, mean_name, std_name = self.ModelFileNames(load_epoch)
        self.model = keras.models.load_model(model_name, custom_objects=loss.LossDict())
        self.mean = np.load(mean_name)
        self.std = np.load(std_name)
    
    def LoadHistory(self):
        self.losses = self.LoadHistoryUnit('loss')
        self.val_losses = self.LoadHistoryUnit('val_loss')
        self.test_increase = self.LoadHistoryUnit('test')
        self.init_epoch = self.MaxModelEpoch()


    def ModelExist(self, epoch=-1):
        temp_path_name, model_name, mean_name, std_name = self.ModelFileNames()
        return (os.path.exists(model_name) and os.path.exists(mean_name) and os.path.exists(std_name))

    def ReshapeRnnFeatures(self, features):
        return features.reshape(features.shape[0], self.feature_unit_num, self.feature_unit_size)

    def FeaturesPretreat(self, features):
        features = (features - self.mean) / self.std
        features = self.ReshapeRnnFeatures(features)
        return features

    def SaveHistory(self, losses, val_losses, test_increase):
        temp_path_name = self.model_path
        if not os.path.exists(temp_path_name):
            os.makedirs(temp_path_name)
        if len(losses) > 0:
            np.save('%s/loss.npy' % temp_path_name, np.array(losses))
        if len(val_losses) > 0:
            np.save('%s/val_loss.npy' % temp_path_name, np.array(val_losses))
        if len(test_increase):
            np.save('%s/test.npy' % temp_path_name, np.array(test_increase))

    def ShowHistory(self):
        path_name = self.model_path
        if not os.path.exists(path_name):
            print("ShowHistory.Error: path (%s) not exist" % path_name)
            return
        
        losses = []
        val_losses = []
        test_increase = []
        train_increase = []

        temp_file_name = '%s/loss.npy' % path_name
        if os.path.exists(temp_file_name):
            losses = np.load(temp_file_name).tolist()
        temp_file_name = '%s/val_loss.npy' % path_name
        if os.path.exists(temp_file_name):
            val_losses = np.load(temp_file_name).tolist()
        temp_file_name = '%s/train.npy' % path_name
        if os.path.exists(temp_file_name):
            train_increase = np.load(temp_file_name).tolist()
        temp_file_name = '%s/test.npy' % path_name
        if os.path.exists(temp_file_name):
            test_increase = np.load(temp_file_name).tolist()

        PlotHistory(path_name, losses, val_losses, train_increase, test_increase)

    def Train(self, train_features, train_labels, val_features, val_labels, train_epochs):
        print("reorder...")
        np.random.seed(0)
        order=np.argsort(np.random.random(len(train_labels)))
        train_features=train_features[order]
        train_labels=train_labels[order]

        print("pretreat...")
        where_are_nan = np.isnan(train_features)
        where_are_inf = np.isinf(train_features)
        train_features[where_are_nan] = 0.0
        train_features[where_are_inf] = 0.0
        self.mean = train_features.mean(axis=0)
        self.std = train_features.std(axis=0)
        train_features = self.FeaturesPretreat(train_features)
        print("train_features: {}".format(train_features.shape))
        val_features = self.FeaturesPretreat(val_features)
        
        if self.continue_train and (self.MaxModelEpoch > 0):
            self.LoadModel()
            self.LoadHistory()
            # self.model.compile()
        else:
            self.model = self.BuildModel()
        
        self.model.summary()

        class TrainCallback(keras.callbacks.Callback):
            def __init__(self, o_dl_model):
                self.dl_model = o_dl_model
                self.losses = o_dl_model.losses
                self.val_losses = o_dl_model.val_losses
                self.test_increase = o_dl_model.test_increase

            def on_epoch_end(self, epoch, logs={}):
                self.dl_model.current_epoch = epoch + 1
                temp_epoch = self.dl_model.current_epoch + self.dl_model.init_epoch
                sys.stdout.write('\r%d' % (temp_epoch))
                sys.stdout.flush()
                self.losses.append([temp_epoch, logs.get('loss')])
                self.val_losses.append([temp_epoch, logs.get('val_loss')])
                if temp_epoch % self.dl_model.save_step == 0:
                    if self.dl_model.test_func != None:
                        self.test_increase.append([temp_epoch, self.dl_model.test_func(self.dl_model.test_param, self.model)])
                    self.dl_model.SaveModel(self.model, mean, std, temp_epoch)
                    self.dl_model.SaveHistory(self.losses, self.val_losses, self.test_increase)

        # The patience parameter is the amount of epochs to check for improvement.
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)
        history = self.model.fit(train_features, train_labels, epochs=train_epochs, batch_size=self.batch_size, 
                            validation_data=(val_features, val_labels), verbose=0,
                            # callbacks=[early_stop, TestCallback()])
                            callbacks=[TrainCallback(self)])

        self.ShowHistory()

    def Predict(self, features, features_pretreated=False):
        if not features_pretreated:
            features = self.FeaturesPretreat(features)
        return self.model.predict(features, batch_size=10240)


    


