# -*- coding:UTF-8 -*-

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
from tensorflow import keras
from compiler.ast import flatten
import numpy as np
import pandas as pd
import time
import sys

reload(sys)
sys.setdefaultencoding('utf-8')
print(tf.__version__)

pd.set_option('display.max_rows', 10)  # 设置显示最大行

import tushare as ts
print(ts.__version__)
ts.set_token('230c446ae448ec95357d0f7e804ddeebc7a51ff340b4e6e0913ea2fa')
pro = ts.pro_api()

feature_file_name='feature_data.npy'
label_file_name='label_data.npy'
if os.path.exists(feature_file_name) and os.path.exists(label_file_name):
  feature_data=np.load(feature_file_name)
  label_data=np.load(label_file_name)
else:
  #下载数据，生成code_list  <<<<<<
  file_name='./data/'+'stock_code'+'.csv'
  if os.path.exists(file_name):
    print("read_csv:%s" % file_name)
    load_df=pd.read_csv(file_name)
  else:
    load_df = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,list_date')
    load_df.to_csv(file_name)
  load_df=load_df[load_df['list_date']<=20100101]
  load_df=load_df[load_df['industry']=='软件服务']
  print(load_df)
  print("\n\n\n")
  code_list=load_df['ts_code'].values
  #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

  #下载数据，生成feature_list和label_list  <<<<<<
  feature_list=[]
  label_list=[]

  #code_list=['600050.SH', '600104.SH']
  current_date=time.strftime('%Y%m%d',time.localtime(time.time()))
  for code_index in range(0, len(code_list)):
      stock_code=code_list[code_index]
      file_name='./data/'+stock_code+'_'+current_date+'.csv'
      if os.path.exists(file_name):
          print("read_csv:%s" % file_name)
          load_df=pd.read_csv(file_name)
      else:
          load_df=pro.daily_basic(ts_code=stock_code, start_date='20100101', end_date=current_date)
          load_df.to_csv(file_name)

      src_df=load_df[['trade_date', 'close', 'turnover_rate']].copy()
      src_df['increase']=0.0

      for iloop in range(0, len(src_df)-1):
          src_df.iloc[iloop,3]=(100*src_df['close'][iloop]/src_df['close'][iloop+1])-100.0
          #src_df['increase'][iloop]=(src_df['close'][iloop]/src_df['close'][iloop-1])-1.0
      print("src_df: {}".format(src_df.shape))
      print(src_df)

      feature_days=10
      for day_loop in range(0, len(src_df)-feature_days-1):
          feature=[]
          for iloop in range(0, feature_days):
              feature.append(src_df['increase'][day_loop+iloop+1])
              feature.append(src_df['turnover_rate'][day_loop+iloop+1])
          feature_list.append(feature)
          label_list.append(src_df['increase'][day_loop])
  #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

  feature_data=np.array(feature_list)
  label_data=np.array(label_list)
  np.save('feature_data.npy', feature_data)
  np.save('label_data.npy', label_data)

print("\nfeature_data[0]:")
print(feature_data[0])

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
    keras.layers.Dense(8, activation=tf.nn.relu, 
                       input_shape=(train_data.shape[1],)),
    keras.layers.Dense(8, activation=tf.nn.relu),
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

EPOCHS = 80

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
