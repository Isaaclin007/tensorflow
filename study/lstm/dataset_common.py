# -*- coding:UTF-8 -*-

import numpy as np
import sys
import math
import time



# 一个 dataset 单元，例如 train 或 test
class DatasetUnit():
    def __init__(self):
        self.num_examples = 0

    def Init(self, features, labels):
        self.num_examples = len(features)
        self.features = features
        self.labels = labels
        self.current_pos = 0
        self.current_order_pos = 0
        self.order = np.argsort(np.random.random(self.num_examples))

    def ResetPos(self):
        self.current_pos = 0

    def NextBatch(self, batch_size, fake_data=False):
        if batch_size == -1:
            return self.features, self.labels
            self.current_pos = 0
        else:
            if self.num_examples < batch_size:
                return [], []
            if self.current_pos == self.num_examples:
                return [], []

            end_pos = self.current_pos + batch_size
            if end_pos > self.num_examples:
                left_num = self.num_examples - self.current_pos
                lack_num = batch_size - left_num
                if (self.current_order_pos + lack_num) > self.num_examples:
                    self.current_order_pos = 0
                batch_order = np.zeros(batch_size, dtype = np.int32)
                batch_order[0:left_num] = range(self.current_pos, self.num_examples)
                batch_order[left_num:] = self.order[self.current_order_pos:self.current_order_pos + lack_num]

                bf = self.features[batch_order]
                bl = self.labels[batch_order]
                self.current_order_pos += lack_num
                self.current_pos = self.num_examples
                return bf, bl

            else:
                bf = self.features[self.current_pos : end_pos]
                bl = self.labels[self.current_pos : end_pos]
                self.current_pos += batch_size
                return bf, bl


