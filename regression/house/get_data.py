#import tushare as ts
#print(ts.__version__)

#temp_data=ts.get_hist_data('600050', start='2003-01-01', end='2018-10-07')
#temp_data=ts.get_h_data('600050', start='2003-01-01', end='2018-10-07')
#print("get_hist_data")
#print(temp_data)
#print("\n")

#temp_data=ts.get_h_data('600050', '2003-01-01', '2013-10-07')
#print("get_h_data")
#print(temp_data)
#print("\n")

import numpy as np

#load_data=np.loadtxt('./data/000001-utf8.txt', delimiter='\t')
load_data=np.loadtxt('./data/000001-.txt')
#load_data=np.loadtxt('./data/000001.txt')
print("000001:")
print("size:" + str(len(load_data)))
print(load_data)
print("\n")

train_data=load_data[0:-1]
print("train_data:")
print("size:" + str(len(train_data)))
print(train_data)
print("\n")

train_label=load_data[1:,0:1]
print("train_label:")
print("size:" + str(len(train_label)))
print(train_label)
print("\n")