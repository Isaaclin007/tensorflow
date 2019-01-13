# -*- coding:UTF-8 -*-

import tensorflow as tf
from tensorflow import keras
import tushare as ts
import numpy as np
import pandas as pd
import os
import time
import sys
import tushare_data

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

train_features, train_labels = tushare_data.GetTrainData()

mean = train_features.mean(axis=0)
std = train_features.std(axis=0)
print("mean: {}".format(mean.shape))
print("std: {}".format(std.shape))
train_features = (train_features - mean) / std

#Create the model
def build_model():
    model = keras.Sequential([
        keras.layers.Dense(32, activation=tf.nn.relu, input_shape=(train_features.shape[1],)),
        keras.layers.Dense(32, activation=tf.nn.relu),
        # keras.layers.Dense(64, activation=tf.nn.relu),
        # keras.layers.Dense(32, activation=tf.nn.relu),
        # keras.layers.Dense(16, activation=tf.nn.relu),
        keras.layers.Dense(1)
    ])

    optimizer = tf.train.RMSPropOptimizer(0.001)

    model.compile(loss='mse',
                    optimizer=optimizer,
                    metrics=['mae'])
    return model

model = build_model()
model.summary()

# Display training progress by printing a single dot for each completed epoch.
class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs):
        if epoch % 1 == 0: print('.')
        #print('.', end='')
        #print('.')

EPOCHS = 20

# The patience parameter is the amount of epochs to check for improvement.
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=200)

history = model.fit(train_features, train_labels, epochs=EPOCHS,
                    validation_split=0.2, verbose=0,
                    #callbacks=[early_stop, PrintDot()])
                    callbacks=[PrintDot()])

history_df=pd.DataFrame(np.array(history.history['val_mean_absolute_error']), columns=['val_err'])
print("\n\n")
print(history_df)
print("\n\n")

# print("%-12s%-12s%-12s" %('epoch', 'train_err', 'val_err'))
# for iloop in history.epoch:
#     train_err=history.history['mean_absolute_error'][iloop]
#     val_err=history.history['val_mean_absolute_error'][iloop]
#     print("%8u%8.2f%8.2f" %(iloop, train_err, val_err))

# 显示 <<<<<<<<<<
import matplotlib.pyplot as plt
def plot_history(history):
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [1000$]')
    plt.plot(history.epoch, np.array(history.history['mean_absolute_error']), 
            label='Train Loss')
    plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']),
            label = 'Val loss')
    plt.legend()
    #plt.ylim([0,5])
    plt.show()
print("\nplot_history")
plot_history(history)
# 显示 >>>>>>>>>>>>

model.save("./model/model.h5")
np.save('./model/mean.npy', mean)
np.save('./model/std.npy', std)
