# -*- coding:UTF-8 -*-

import tensorflow as tf
import numpy as np
import sys
sys.path.append("../../regression")
from common import base_common
from common import np_common
import math
import time
import dataset_common

file_name_mnist_images = './data/images.npy'
file_name_mnist_labels = './data/labels.npy'

def CreateDataset():
    mnist=tf.keras.datasets.mnist

    (x_train,y_train), (x_test,y_test)=mnist.load_data()
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)
    images = np.zeros((x_train.shape[0] + x_test.shape[0], x_train.shape[1], x_train.shape[2]))
    images[:x_train.shape[0]] = x_train[:]
    images[x_train.shape[0]:] = x_test[:]
    print(images.shape)
    np.save(file_name_mnist_images, images)

    labels = np.zeros((y_train.shape[0] + y_test.shape[0], ))
    labels[:y_train.shape[0]] = y_train[:]
    labels[y_train.shape[0]:] = y_test[:]
    print(labels.shape)
    np.save(file_name_mnist_labels, labels)

def GetDataset(val_ratio, flatten_image=False, one_hot=False):
    images = np.load(file_name_mnist_images).astype(np.float32) / 256.0
    labels = np.load(file_name_mnist_labels).astype(np.float32)

    if flatten_image:
        images = images.reshape((images.shape[0], images.shape[1] * images.shape[2]))
    if one_hot:
        labels = np.eye(10).astype(np.float32)[labels.astype(np.int32)]
    train_num = int(len(images) * (1.0 - val_ratio))
    print('dtype: {}, {}'.format(images.dtype, labels.dtype))
    tf = images[:train_num]
    vf = images[train_num:]
    tl = labels[:train_num]
    vl = labels[train_num:]
    print('train features : {}'.format(tf.shape))
    print('train labels   : {}'.format(tl.shape))
    print('val features   : {}'.format(vf.shape))
    print('val labels     : {}'.format(vl.shape))
    return tf, tl, vf, vl


class MnistDataset():
    def __init__(self, val_ratio, test_ratio, flatten_image=False, one_hot=False):
        images = np.load(file_name_mnist_images).astype(np.float32) / 256.0
        labels = np.load(file_name_mnist_labels).astype(np.float32)

        if flatten_image:
            images = images.reshape((images.shape[0], images.shape[1] * images.shape[2]))
        if one_hot:
            labels = np.eye(10).astype(np.float32)[labels.astype(np.int32)]

        data_num = len(images)
        val_num = int(data_num * val_ratio)
        test_num = int(data_num * test_ratio)

        p1 = 0
        p2 = val_num
        self.validation = dataset_common.DatasetUnit()
        self.validation.Init(images[p1:p2], labels[p1:p2])

        p1 = val_num
        p2 = val_num + test_num
        self.test = dataset_common.DatasetUnit()
        self.test.Init(images[p1:p2], labels[p1:p2])

        p1 = val_num + test_num
        p2 = data_num
        self.train = dataset_common.DatasetUnit()
        self.train.Init(images[p1:p2], labels[p1:p2])

class TestSeqDataset():
    def __init__(self, data_num, step_num, val_ratio, test_ratio, flatten=False, one_hot=False):
        FEATURE_D_VALUE = True
        FEATURE_COMP_RAND = True
        np.random.seed(123)
        seq_data = np.random.ranf([data_num + step_num + 100, 10])
        # h = 0.7
        # for i in range(0, len(seq_data)):
        #     seq_data[i][0] = (h + 0.4) ** (seq_data[i][1] + 0.5)
        #     h = seq_data[i][0] - h
        #     if h < 0:
        #         h = 0.1
        # show_data = np.zeros([len(seq_data), 2])
        # for i in range(0, len(show_data)):
        #     show_data[i][0] = i
        #     show_data[i][1] = seq_data[i][0]
        # np_common.Show2DData('SeqDataset', [show_data])
        # np_common.ShowHist(seq_data[:, 0], 0.01)
        for i in range(0, len(seq_data)):
            seq_data[i][0] = math.sin(i / 10.0)
            seq_data[i][1] = math.sin(i / 9.0)
            if FEATURE_COMP_RAND:
                seq_data[i][0] += (seq_data[i][6] - 0.5) / 100.0
                seq_data[i][1] += (seq_data[i][7] - 0.5) / 100.0
            if i >= 1:
                seq_data[i][2] = seq_data[i][0] - seq_data[i - 1][0]
                seq_data[i][3] = seq_data[i][1] - seq_data[i - 1][1]
                seq_data[i][4] = seq_data[i][2] + seq_data[i][3]
        # show_data = np.zeros([len(seq_data), 2])
        # for i in range(0, len(show_data)):
        #     show_data[i][0] = i / 10.0
        #     show_data[i][1] = seq_data[i][0]
        # np_common.Show2DData('SeqDataset', [show_data])

        features = np.zeros([data_num, step_num, 2])
        labels = np.zeros([data_num,])
        for i in range(0, data_num):
            seq_index = i + 50
            if FEATURE_D_VALUE:
                features[i] = seq_data[seq_index : seq_index + step_num, 2:4]
            else:
                features[i] = seq_data[seq_index : seq_index + step_num, 0:2]
            labels[i] = seq_data[seq_index + step_num][4] * 10.0
        # np_common.ShowHist(labels, 0.1)

        np.random.seed(456)
        order = np.argsort(np.random.random(data_num))
        features = features[order]
        labels = labels[order]

        if flatten:
            features = features.reshape((features.shape[0], features.shape[1] * features.shape[2]))
        if one_hot:
            labels = np.eye(2).astype(np.float32)[(labels > 0).astype(np.int32)]

        # np_common.ShowHist(labels, 1)

        # print(features)
        # print(labels)

        val_num = int(data_num * val_ratio)
        test_num = int(data_num * test_ratio)

        p1 = 0
        p2 = val_num
        self.validation = dataset_common.DatasetUnit()
        self.validation.Init(features[p1:p2], labels[p1:p2])

        p1 = val_num
        p2 = val_num + test_num
        self.test = dataset_common.DatasetUnit()
        self.test.Init(features[p1:p2], labels[p1:p2])

        p1 = val_num + test_num
        p2 = data_num
        self.train = dataset_common.DatasetUnit()
        self.train.Init(features[p1:p2], labels[p1:p2])

# seq_dataset = TestSeqDataset(10000, 10, 0.1, 0.1, True, True)

# print('start')
# for i in range(1):
#     while True:
#         x, y = seq_dataset.train.NextBatch(1024)
#         if (len(x) == 0):
#             seq_dataset.train.Reset()
#             break
#         # print('{}, {}'.format(x.shape, y.shape))
