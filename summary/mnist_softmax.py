# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A very simple MNIST classifier.
See extensive documentation at
https://www.tensorflow.org/get_started/mnist/beginners
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf



def main(_):
  mnist = tf.keras.datasets.mnist
  (x_train_mat, y_train),(x_test_mat, y_test) = mnist.load_data()
  x_train = x_train_mat.reshape((len(x_train_mat),784, ))
  x_test = x_test_mat.reshape((len(x_test_mat), 784, ))
  print(type(x_train))
  print(x_train.shape)
  print(y_train.shape)
  print(x_test.shape)
  print(y_test.shape)

  # np.set_printoptions(linewidth=150)
  # print(x_train[0].reshape((28,28)))

  # Create the model
  x = tf.placeholder(tf.float32, [None, 784])
  W = tf.Variable(tf.zeros([784, 10]))
  b = tf.Variable(tf.zeros([10]))
  y = tf.matmul(x, W) + b

  # Define loss and optimizer
  y_ = tf.placeholder(tf.int64, [None])
  print(y)
  print(y_)

  # The raw formulation of cross-entropy,
  #
  #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
  #                                 reduction_indices=[1]))
  #
  # can be numerically unstable.
  #
  # So here we use tf.losses.sparse_softmax_cross_entropy on the raw
  # outputs of 'y', and then average across the batch.
  cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=y)
  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

  correct_prediction = tf.equal(tf.argmax(y, 1), y_)
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  tf.summary.scalar('accuracy', accuracy)
  merged = tf.summary.merge_all()

  sess = tf.InteractiveSession()
  train_writer = tf.summary.FileWriter('./log/mnist_softmax', sess.graph)

  tf.global_variables_initializer().run()
  # Train
  batch_size = 100
  batch_num = int(len(x_train) / batch_size)
  for iloop in range(batch_num):
    offset = iloop * batch_size
    batch_xs = x_train[offset:offset+batch_size]
    batch_ys = y_train[offset:offset+batch_size]
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    if (iloop % 10) == 0:
      # Test trained model
      summary, acc = sess.run([merged, accuracy], feed_dict={x: x_test, y_: y_test})
      train_writer.add_summary(summary, iloop)
      print(acc)

if __name__ == '__main__':
  tf.app.run(main=main, argv=[sys.argv[0]])
