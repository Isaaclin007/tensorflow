# -*- coding:UTF-8 -*-

import tensorflow as tf
from tensorflow import keras
import tushare as ts
import numpy as np
import pandas as pd
import os
import time
import sys

reload(sys)
sys.setdefaultencoding('utf-8')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

#pd.set_option('display.width', 150)  # 设置字符显示宽度
#pd.set_option('display.max_rows', None)  # 设置显示最大行

print("load...")
train_data=np.load("train_data.npy")
print("train_data: {}".format(train_data.shape))

print("reorder...")
order=np.argsort(np.random.random(len(train_data)))
train_data=train_data[order]
train_data=train_data[:300000]

col_num=train_data.shape[1]
train_features=train_data[:,0:col_num-2]
#train_labels=train_data[:,col_num-2:col_num-1]
train_labels=train_data[:,col_num-1:]
print("train_features: {}".format(train_features.shape))
print("train_labels: {}".format(train_labels.shape))

mean = train_features.mean(axis=0)
std = train_features.std(axis=0)
print("mean: {}".format(mean.shape))
print("std: {}".format(std.shape))
train_features = (train_features - mean) / std

#Create the model
def build_model():
    model = keras.Sequential([
        keras.layers.Dense(16, activation=tf.nn.relu, input_shape=(train_features.shape[1],)),
        keras.layers.Dense(16, activation=tf.nn.relu),
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

EPOCHS = 300

# The patience parameter is the amount of epochs to check for improvement.
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=200)

history = model.fit(train_features, train_labels, epochs=EPOCHS,
                    validation_split=0.2, verbose=0,
                    #callbacks=[early_stop, PrintDot()])
                    callbacks=[PrintDot()])

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

model.save("model.h5")