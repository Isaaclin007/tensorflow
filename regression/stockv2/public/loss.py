# -*- coding:UTF-8 -*-


import numpy as np
import os
import sys
import math
import random
from absl import app
from absl import flags
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import backend as K
from tensorflow.python.ops import math_ops

FLAGS = flags.FLAGS
flags.DEFINE_string('loss', 'mae', 'loss func name')

# x:[-2 ~ 2], y:[-0.964 ~ 0.964]
def Tanh(x):
    return K.tanh(x)

def TanhTest():
    x = np.arange(-2, 2.1, 0.1) 
    y_op = Tanh(x)
    
    with tf.Session() as sess:
        y = y_op.eval()
    
    show_data = np.zeros((len(x), 2))
    show_data[:, 0] = x
    show_data[:, 1] = y
    print(show_data)
    # np_common.Show2DData('Tanh', [show_data], [])

def LossAbs(y_true, y_pred, e=0.1):
    return K.abs(y_true - y_pred)

def LossTanhDiff(y_true, y_pred, e=0.1):
    return K.abs(K.tanh((y_true - 5.0) * 0.4) - K.tanh((y_pred - 5.0) * 0.4))

def LossTP0MaxRatio(y_true, y_pred, e=0.1):
    return K.abs(y_true - y_pred) / 10.0 * K.maximum(K.maximum(y_true, y_pred), 0.0)

def LossTP0MaxP1MaxRatio(y_true, y_pred, e=0.1):
    return K.abs(y_true - y_pred) * K.maximum(K.maximum(y_true, y_pred), 0.0) * K.maximum(y_pred, 1.0) * 0.1

def LossTanhDiffTP0MaxRatio(y_true, y_pred, e=0.1):
    true_map = (K.tanh((y_true - 5.0) * 0.4) + 1.00001) * 5.0  # 0 ~ 10
    pred_map = (K.tanh((y_pred - 5.0) * 0.4) + 1.00001) * 5.0
    # return abs(true_map - pred_map) * (pred_map + 1.0)
    return K.abs(true_map - pred_map) * K.maximum(K.maximum(y_true, y_pred), 0.0)

def LossTanhDiffP1MaxRatio(y_true, y_pred, e=0.1):
    true_map = (K.tanh((y_true - 5.0) * 0.4) + 1.00001) * 5.0  # 0 ~ 10
    pred_map = (K.tanh((y_pred - 5.0) * 0.4) + 1.00001) * 5.0
    return K.abs(true_map - pred_map) * K.maximum(y_pred, 1.0)

def mean_absolute_error(y_true, y_pred):
    return K.mean(math_ops.abs(y_pred - y_true), axis=-1)

def absolute_tp0_max_ratio_error(y_true, y_pred):
    return math_ops.abs(y_true - y_pred) * math_ops.maximum(math_ops.maximum(y_true, y_pred), 0.0)

def mean_absolute_tp0_max_ratio_error(y_true, y_pred):
    return K.mean(math_ops.abs(y_true - y_pred) * math_ops.maximum(math_ops.maximum(y_true, y_pred), 0.0))

def mean_absolute_tp_max_ratio_error_tanhmap(y_true, y_pred):
    # y:[0 ~ 10], map:[0.2 ~ 9.8]
    # y < 0     , map:[0 ~ 0.2]
    # y > 10    , map:[9.8 ~ 10]
    t_map = (K.tanh((y_true - 5.0) * 0.4) + 1.00001) * 5.0
    p_map = (K.tanh((y_pred - 5.0) * 0.4) + 1.00001) * 5.0
    return K.mean(math_ops.abs(t_map - p_map) * math_ops.maximum(t_map, p_map))

loss_dict = {'LossAbs': LossAbs,
             'LossTanhDiff': LossTanhDiff,
             'LossTP0MaxRatio': LossTP0MaxRatio,
             'LossTP0MaxP1MaxRatio': LossTP0MaxP1MaxRatio,
             'LossTanhDiffTP0MaxRatio': LossTanhDiffTP0MaxRatio,
             'LossTanhDiffP1MaxRatio': LossTanhDiffP1MaxRatio,
             'mean_absolute_error':mean_absolute_error,
             'mean_absolute_tp0_max_ratio_error':mean_absolute_tp0_max_ratio_error,
             'mean_absolute_tp_max_ratio_error_tanhmap':mean_absolute_tp_max_ratio_error_tanhmap}

def LossDict():
    return loss_dict

def LossFunc(func_name):
    return loss_dict[func_name]


    
    




def main(argv):
    del argv

    # y_true = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 100.0, 0.0,  0.0, 10.0, 10.0, -10.0])
    # y_pred = np.array([-1.5, 0.5, 1.5, 2.5, 3.5, 100.5, 10.0, 0.0, 0.0, -10.0,  10.0])
    # print(y_true)
    # print(y_pred)
    # print('')
    # print(FLAGS.loss)
    # print('-' * 40)
    # temp_loss = LossFunc(FLAGS.loss)(y_true, y_pred)
    # with tf.Session() as sess:
    #     print(temp_loss.eval())

    TanhTest()
    exit()

if __name__ == "__main__":
    app.run(main)
    


