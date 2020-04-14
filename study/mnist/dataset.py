# -*- coding:UTF-8 -*-

import tensorflow as tf
import numpy as np

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



# 一个 dataset 单元，例如 train 或 test
class DatasetUnit():
    def __init__(self):
        self.num_examples = 0

    def init(self, features, labels):
        self.num_examples = len(features)
        self.features = features
        self.labels = labels
        self.current_pos = 0

    def next_batch(self, batch_size, fake_data=False):
        if batch_size == -1:
            return self.features, self.labels
            self.current_pos = 0
        else:
            if self.num_examples < batch_size:
                return [], []
            end_pos = self.current_pos + batch_size
            if end_pos > self.num_examples:
                self.current_pos = 0
                return self.next_batch(batch_size, fake_data)
            else:
                bf = self.features[self.current_pos : end_pos]
                bl = self.labels[self.current_pos : end_pos]
                self.current_pos += batch_size
                return bf, bl

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
        self.validation = DatasetUnit()
        self.validation.init(images[p1:p2], labels[p1:p2])

        p1 = val_num
        p2 = val_num + test_num
        self.test = DatasetUnit()
        self.test.init(images[p1:p2], labels[p1:p2])

        p1 = val_num + test_num
        p2 = data_num
        self.train = DatasetUnit()
        self.train.init(images[p1:p2], labels[p1:p2])
