# -*- coding:UTF-8 -*-

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
from tensorflow import keras
from compiler.ast import flatten
import numpy as np
print(tf.__version__)

print("\nLoad...")
load_data=np.loadtxt('./data/600050w.txt', skiprows=30, usecols=(1,2,3,4,5,6,7))
print("load_data: {}".format(load_data.shape))

#过去10天的数据作为特征，数量是load_data行数-10
feature_data=np.ndarray(shape=(len(load_data)-10,41))
row_index=0
for day_loop in range(0, len(load_data)-10):
  col_index=0
  feature_data[row_index, col_index]=day_loop
  col_index=col_index+1
  for iloop in range(0, 10):
    feature_data[row_index, col_index]=load_data[day_loop+iloop][3] #收盘价
    col_index=col_index+1
    feature_data[row_index, col_index]=load_data[day_loop+iloop][4] #涨幅
    col_index=col_index+1
    feature_data[row_index, col_index]=load_data[day_loop+iloop][5] #振幅
    col_index=col_index+1
    feature_data[row_index, col_index]=load_data[day_loop+iloop][6] #交易量
    col_index=col_index+1
  row_index=row_index+1

#10-倒数第一天的价格，数量是load_data行数-10
label_data=load_data[10:,4]
#label_data.shape=len(label_data)
print("label_data: {}".format(label_data.shape))

#将feature和label划分为训练集和测试集
print("\nCreate train data and test data ...")
train_data_num=len(label_data)-5
train_data=feature_data[:train_data_num]
train_labels=label_data[:train_data_num]
test_data=feature_data[train_data_num:]
test_labels=label_data[train_data_num:]
print("Training set: {}".format(train_data.shape))
print("Testing set:  {}".format(test_data.shape)) 
print("Training labels: {}".format(train_labels.shape))
print("Testing labels:  {}".format(test_labels.shape)) 

#feature和label按照相同的随机序列重新排列
print("\nReorder...")
order=np.argsort(np.random.random(train_labels.shape))
train_data=train_data[order]
train_labels=train_labels[order]


#显示数据
""" print("Show data ...")
import pandas as pd
column_names = ['开盘', '最高', '最低', '收盘', '涨跌', '涨幅', '振幅', '总手', '金额']
df = pd.DataFrame(train_data, columns=column_names)
df.head()
print("\ntrain_labels[0:10]:")
print(train_labels[0:10]) """

#Test data is *not* used when calculating the mean and std.
mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
train_data = (train_data - mean) / std
test_data = (test_data - mean) / std
print("\ntrain_data[0]:")
print(train_data[0])  # First training sample, normalized

#Create the model
def build_model():
  model = keras.Sequential([
    keras.layers.Dense(16, activation=tf.nn.relu, 
                       input_shape=(train_data.shape[1],)),
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
    if epoch % 100 == 0: print('.')
    #print('.', end='')
    #print('.')

EPOCHS = 200

# The patience parameter is the amount of epochs to check for improvement.
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=200)

history = model.fit(train_data, train_labels, epochs=EPOCHS,
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


test_predictions = model.predict(test_data).flatten()
print("\ntest_predictions")
print(test_predictions)
