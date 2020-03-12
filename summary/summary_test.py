from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import tensorflow as tf

with tf.name_scope('bbb'):
  var_1 = tf.Variable(initial_value=[0.0], name='var_1')

# with tf.variable_scope('aaa', reuse=True):
with tf.variable_scope('aaa', reuse=tf.AUTO_REUSE):
  var_1 = tf.Variable(initial_value=[0.0], name='var_1')
  var_2 = tf.get_variable(name='var_1', shape=[1,])
  var_3 = tf.get_variable(name='var_1', shape=[1,])

print(var_1)
print(var_2)
print(var_3)