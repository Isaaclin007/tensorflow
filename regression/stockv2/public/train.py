# -*- coding:UTF-8 -*-


import numpy as np
import os
import time
import datetime
import sys
import math
import random
import tensorflow as tf
import lstm_model
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def GetDatasetSplitByDate(file_name, split_date, one_hot=False):
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


    if one_hot:
        train_labels = np.eye(2).astype(np.float32)[(train_labels > 0).astype(np.int32)]
        val_labels = np.eye(2).astype(np.float32)[(val_labels > 0).astype(np.int32)]

    return train_features, train_labels, val_features, val_labels

def GetDatasetSplitRandom(file_name, val_ratio, one_hot=False):
    dataset = np.load(file_name)
    print("dataset: {}".format(dataset.shape))
    data_len = len(dataset)
    val_data_len = int(data_len * val_ratio)
    np.random.seed(123)
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

    if one_hot:
        train_labels = np.eye(2).astype(np.float32)[(train_labels > 0).astype(np.int32)]
        val_labels = np.eye(2).astype(np.float32)[(val_labels > 0).astype(np.int32)]

    return train_features, train_labels, val_features, val_labels

def FeaturesPretreat(features, mean, std, feature_unit_num, feature_unit_size):
    features = (features - mean) / std
    output_shape = []
    for iloop in range(features.ndim - 1):
        output_shape.append(features.shape[iloop])
    output_shape.append(feature_unit_num)
    output_shape.append(feature_unit_size)
    return features.reshape(output_shape)

if __name__ == "__main__":
    data_split_mode = 'split_by_date'
    if len(sys.argv) >= 2:
        data_split_mode = sys.argv[1]

    feature_unit_num = 7
    feature_unit_size = 5
    file_name = './data/dataset.npy'

    if data_split_mode == 'split_random':
        # 随机抽取指定比例的数据作为验证集，其他作为训练集
        tf_, tl, vf, vl = GetDatasetSplitRandom(file_name, 0.5, False)
    elif data_split_mode == 'split_by_date':
        # 根据数据时间切分训练集和验证集，数据时间大于split_date的作为验证集，其他作为训练集
        tf_, tl, vf, vl = GetDatasetSplitByDate(file_name, 20190101, False)
    else:
        exit()

    mean = tf_.mean(axis=0)
    std = tf_.std(axis=0)
    std[std < 0.0001] = 0.0001
    tf_ = FeaturesPretreat(tf_, mean, std, feature_unit_num, feature_unit_size)
    vf = FeaturesPretreat(vf, mean, std, feature_unit_num, feature_unit_size)

    

    training = lstm_model.TFLstm(batch_size=64, 
                                    num_steps=feature_unit_num, 
                                    vec_size=feature_unit_size,
                                    classify=False,
                                    num_classes=2, 
                                    lstm_size=4, 
                                    lstm_layers_num=1,
                                    learning_rate=0.001,
                                    keep_prob=0.5,
                                    grad_clip=5, 
                                    checkpoint_dir='./ckpt',
                                    log_dir='./ckpt',
                                    continue_train=True)

    training.Fit(tf_, tl, vf, vl, 100)


    
    


