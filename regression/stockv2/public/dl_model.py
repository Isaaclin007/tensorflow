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
from tensorflow.python.keras.callbacks import LearningRateScheduler
import loss
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def ImportMatPlot(use_agg=False):
    import matplotlib
    font_name = r"/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"
    if use_agg:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdate
    from matplotlib.font_manager import FontProperties
    zhfont = FontProperties(fname=font_name, size=15)
    return plt, mdate, zhfont

def RmDir(path):
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):
            RmDir(c_path)
        else:
            os.remove(c_path)
    os.rmdir(path)

def Plot2DArray(ax, arr, name, color=''):
    np_arr = np.array(arr)
    if len(np_arr) > 1:
        x = np_arr[:,0]
        y = np_arr[:,1]
        if color != '':
            ax.plot(x, y, color, label=name)
        else:
            ax.plot(x, y, label=name)

MAX_TEST_FUNCS = 2
# TFR: test funcs results
def PlotHistory(save_path, losses, val_losses, TFR):
    plt, mdate, zhfont = ImportMatPlot(True)
    if len(losses) <= 1:
        return
    plt.ion()
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    Plot2DArray(ax1, losses, 'loss')
    Plot2DArray(ax1, val_losses, 'val_loss')
    for iloop in range(len(TFR)):
        Plot2DArray(ax2, TFR[iloop], 'TFR_%u' % iloop)
    ax1.legend()
    # ax2.legend()
    # plt.show()
    # plt.pause(1)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig('%s/figure.png' % save_path)
    plt.close()


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
                 test_funcs_list = None,
                 test_params_list = None):
        # self.setting_name = '%s_%u_%u_%u_%u_%f_%s' % (app_setting_name, 
        #                                               feature_unit_num, 
        #                                               feature_unit_size, 
        #                                               lstm_size, 
        #                                               batch_size, 
        #                                               learning_rate, 
        #                                               loss)
        self.setting_name = '0'
        self.feature_unit_num = feature_unit_num
        self.feature_unit_size = feature_unit_size
        self.lstm_size = lstm_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.loss = loss
        self.model_path = './model/%s' % self.setting_name
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        self.save_step = save_step
        self.test_funcs_list = test_funcs_list
        self.test_params_list = test_params_list
        self.continue_train = False
        self.init_epoch = 0
        self.losses = []
        self.val_losses = []
        self.TFR = []
        for iloop in range(MAX_TEST_FUNCS):
            self.TFR.append([])

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

        # my_optimizer = keras.optimizers.RMSprop(lr=self.learning_rate, rho=0.9, epsilon=1e-06, decay=0.00005)
        my_optimizer = keras.optimizers.RMSprop(lr=self.learning_rate, rho=0.9, epsilon=1e-06)
        active_loss = loss.LossFunc(self.loss)

        model.compile(loss=active_loss, optimizer=my_optimizer, metrics=[active_loss])
        return model

    def SaveModel(self, model, epoch=-1):
        temp_path_name, model_name, mean_name, std_name = self.ModelFileNames(epoch)
        if not os.path.exists(temp_path_name):
            os.makedirs(temp_path_name)
        model.save(model_name)
        np.save(mean_name, self.mean)
        np.save(std_name, self.std)

    def LoadHistoryUnit(self, his_name):
        file_name = '%s/%s.npy' % (self.model_path, his_name)
        if os.path.exists(file_name):
            return np.load(file_name).tolist()
        else:
            return []
    
    def SaveHistoryUnit(self, his_name, his_data):
        if len(his_data) > 0:
            file_name = '%s/%s.npy' % (self.model_path, his_name)
            np.save(file_name, np.array(his_data))

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
        print('LoadModel:%s' % model_name)
        self.model = keras.models.load_model(model_name, custom_objects=loss.LossDict())
        self.mean = np.load(mean_name)
        self.std = np.load(std_name)
    
    def LoadHistory(self):
        self.losses = self.LoadHistoryUnit('loss')
        self.val_losses = self.LoadHistoryUnit('val_loss')
        self.TFR = []
        for iloop in range(MAX_TEST_FUNCS):
            self.TFR.append(self.LoadHistoryUnit('TFR_%u' % iloop))
        self.init_epoch = self.MaxModelEpoch()


    def ModelExist(self, epoch=-1):
        temp_path_name, model_name, mean_name, std_name = self.ModelFileNames()
        return (os.path.exists(model_name) and os.path.exists(mean_name) and os.path.exists(std_name))

    def ReshapeRnnFeatures(self, features):
        output_shape = []
        for iloop in range(features.ndim - 1):
            output_shape.append(features.shape[iloop])
        output_shape.append(self.feature_unit_num)
        output_shape.append(self.feature_unit_size)
        return features.reshape(output_shape)

    def FeaturesPretreat(self, features):
        # features = (features - self.mean) / self.std
        features = self.ReshapeRnnFeatures(features)
        return features

    def SaveHistory(self, losses, val_losses, TFR):
        self.SaveHistoryUnit('loss', losses)
        self.SaveHistoryUnit('val_loss', val_losses)
        for iloop in range(MAX_TEST_FUNCS):
            self.SaveHistoryUnit('TFR_%u' % iloop, TFR[iloop])    

    def ShowHistory(self):
        path_name = self.model_path
        if not os.path.exists(path_name):
            print("ShowHistory.Error: path (%s) not exist" % path_name)
            return
        losses = self.LoadHistoryUnit('loss')
        val_losses = self.LoadHistoryUnit('val_loss')
        TFR = []
        for iloop in range(MAX_TEST_FUNCS):
            TFR.append(self.LoadHistoryUnit('TFR_%u' % iloop))
        PlotHistory(path_name, losses, val_losses, TFR)

    def Clean(self):
        path_name = self.model_path
        if os.path.exists(path_name):
            RmDir(path_name)

    def Train(self, train_features, train_labels, val_features, val_labels, train_epochs):
        print("reorder...")
        np.random.seed(0)
        order = np.argsort(np.random.random(len(train_labels)))
        train_features = train_features[order]
        train_labels = train_labels[order]

        print("pretreat...")
        where_are_nan = np.isnan(train_features)
        where_are_inf = np.isinf(train_features)
        train_features[where_are_nan] = 0.0
        train_features[where_are_inf] = 0.0
        self.mean = train_features.mean(axis=0)
        self.std = train_features.std(axis=0)
        self.std[self.std < 0.0001] = 0.0001
        train_features = self.FeaturesPretreat(train_features)
        print("train_features: {}".format(train_features.shape))
        val_features = self.FeaturesPretreat(val_features)
        
        if self.continue_train and (self.MaxModelEpoch() > 0):
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
                self.TFR = o_dl_model.TFR

            def run_test_funcs(self):
                if self.dl_model.test_funcs_list != None:
                    for iloop in range(len(self.dl_model.test_funcs_list)):
                        test_func = self.dl_model.test_funcs_list[iloop]
                        test_param = self.dl_model.test_params_list[iloop]
                        test_result = test_func(test_param, self.model)
                        self.TFR[iloop].append(test_result)

            def on_epoch_end(self, epoch, logs={}):
                self.dl_model.current_epoch = epoch + 1
                temp_epoch = self.dl_model.current_epoch + self.dl_model.init_epoch
                sys.stdout.write('\r%d' % (temp_epoch))
                sys.stdout.flush()
                self.losses.append([temp_epoch, logs.get('loss')])
                self.val_losses.append([temp_epoch, logs.get('val_loss')])
                if temp_epoch % self.dl_model.save_step == 0:
                    self.run_test_funcs()
                    self.dl_model.SaveModel(self.model, temp_epoch)
                    self.dl_model.SaveHistory(self.losses, self.val_losses, self.TFR)
                    self.dl_model.ShowHistory()

        def LrScheduler(epoch):
            # 每隔100个epoch，学习率减小为原来的1/10
            # if epoch % 1 == 0 and epoch != 0:
            #     lr = K.get_value(self.model.optimizer.lr)
            #     K.set_value(self.model.optimizer.lr, lr * 0.98)
            if epoch % 100 == 0 and epoch != 0:
                # print
                lr = K.get_value(self.model.optimizer.lr)
                print("\nlr: {}".format(lr))
            return K.get_value(self.model.optimizer.lr)
 
        reduce_lr = LearningRateScheduler(LrScheduler)

        # The patience parameter is the amount of epochs to check for improvement.
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)
        history = self.model.fit(train_features,
                                 train_labels, 
                                 epochs=train_epochs, 
                                 batch_size=self.batch_size, 
                                 validation_data=(val_features, val_labels), 
                                 verbose=0,
                                 callbacks=[TrainCallback(self), reduce_lr])

        self.ShowHistory()

    def Predict(self, features, features_pretreated=False):
        if not features_pretreated:
            features = self.FeaturesPretreat(features)
        feature_shape = [1]
        output_prediction_shape = []
        for iloop in range(features.ndim - 2):
            feature_shape[0] *= features.shape[iloop]
            output_prediction_shape.append(features.shape[iloop])
        feature_shape.append(features.shape[features.ndim - 2])
        feature_shape.append(features.shape[features.ndim - 1])
        output_prediction_shape.append(1)
        input_features = features.reshape(feature_shape)
        print('features:{}'.format(input_features.shape))
        predictions = self.model.predict(input_features, batch_size=10240)
        predictions = predictions.reshape(output_prediction_shape)
        return predictions


def GetDatasetSplitByDate(file_name, split_date):
    dataset = np.load(file_name)
    print("dataset: {}".format(dataset.shape))
    pos = dataset[:, -1] < split_date
    train_data = dataset[pos]
    val_data = dataset[~pos]

    print("train: {}".format(train_data.shape))
    print("val: {}".format(val_data.shape))

    feature_size = dataset.shape[1] - 2
    train_features = train_data[:, :feature_size]
    train_labels = train_data[:, feature_size]

    val_features = val_data[:, :feature_size]
    val_labels = val_data[:, feature_size]

    return train_features, train_labels, val_features, val_labels

def GetDatasetSplitRandom(file_name, val_ratio):
    dataset = np.load(file_name)
    print("dataset: {}".format(dataset.shape))
    data_len = len(dataset)
    val_data_len = int(data_len * val_ratio)
    np.random.seed(0)
    order = np.argsort(np.random.random(data_len))
    train_data = dataset[order[val_data_len:]]
    val_data = dataset[order[:val_data_len]]

    feature_size = dataset.shape[1] - 2
    print("train: {}".format(train_data.shape))
    print("val: {}".format(val_data.shape))

    train_features = train_data[:, :feature_size]
    train_labels = train_data[:, feature_size]

    val_features = val_data[:, :feature_size]
    val_labels = val_data[:, feature_size]

    return train_features, train_labels, val_features, val_labels

if __name__ == "__main__":
    data_split_mode = 'split_by_date'
    if len(sys.argv) >= 2:
        data_split_mode = sys.argv[1]

    feature_unit_num = 5
    feature_unit_size = 5
    file_name = './data/dataset.npy'

    if data_split_mode == 'split_random':
        # 随机抽取指定比例的数据作为验证集，其他作为训练集
        tf, tl, vf, vl = GetDatasetSplitRandom(file_name, 0.5)
    elif data_split_mode == 'split_by_date':
        # 根据数据时间切分训练集和验证集，数据时间大于split_date的作为验证集，其他作为训练集
        tf, tl, vf, vl = GetDatasetSplitByDate(file_name, 20100101)
    else:
        exit()

    feature_size = feature_unit_size * feature_unit_num
    origin_feature_size = tf.shape[1]
    if feature_size < origin_feature_size:
        temp_index = origin_feature_size - feature_size
        tf = tf[:, temp_index:].copy()
        vf = vf[:, temp_index:].copy()
        print("train feature: {}".format(tf.shape))
        print("val feature: {}".format(vf.shape))

    o_dl_model = DLModel('',
                         feature_unit_num, 
                         feature_unit_size,
                         64, 
                         10240, 
                         0.01, 
                         'mean_absolute_tp_max_ratio_error_tanhmap_0_15',
                         50)
    start_time = time.time()
    o_dl_model.Train(tf, tl, vf, vl, 250)
    print('\n\nrun time: {}'.format(time.time() - start_time))
    
    


