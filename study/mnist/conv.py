# -*- coding:UTF-8 -*-

import tensorflow as tf
import numpy as np
import os
import dataset

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)  # -0.2 ~ 0.2 正太分布
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)  # 常量
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

x = tf.placeholder("float32", shape=[None, 784])
y_ = tf.placeholder("float32", shape=[None, 10])
x_image = tf.reshape(x, [-1,28,28,1])

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

ti, tl, vi, vl = dataset.GetDataset(0.1, True, True)

batch_size = 32
epoch = 10
print('accuracy:')
print sess.run(accuracy, feed_dict={x: vi, y_: vl, keep_prob: 1.0})
for e in range(epoch):
    for i in range(len(ti) / batch_size):
        batch_start = i * batch_size
        batch_end = batch_start + batch_size
        batch_images = ti[batch_start:batch_end]
        batch_labels = tl[batch_start:batch_end]
        sess.run(train_step, feed_dict={x: batch_images, y_: batch_labels, keep_prob: 0.5})
    print sess.run(accuracy, feed_dict={x: vi, y_: vl, keep_prob: 1.0})

exit()



# tfe = tf.contrib.eager
# tfe.enable_eager_execution()

x_shape = [1, 3, 2, 1]
f_shape = [2, 2, 1, 1]

x = tf.placeholder("float64", shape=x_shape)
f = tf.placeholder("float64", shape=f_shape)

op_conv = tf.nn.conv2d(x, f, strides=[1, 1, 1, 1],
                use_cudnn_on_gpu=False, padding='SAME')

op_relu = tf.nn.relu(op_conv)

op_pool = tf.nn.max_pool(op_relu, ksize=[1, 2, 2, 1],
                strides=[1, 1, 2, 1], padding='VALID')
print(op_conv)
print(op_pool)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
x_data = np.array([-1, -2, -5, -1, -1, -1]).reshape(x_shape).astype(np.float64)
f_data = np.ones(f_shape).astype(np.float64)
print sess.run(op_pool, feed_dict={x: x_data, f: f_data})
exit()







x = tf.placeholder("float64", shape=[None, 784])
y_ = tf.placeholder("float64", shape=[None, 10])

w = tf.Variable(tf.zeros([784, 10], dtype=tf.float64))
b = tf.Variable(tf.zeros([10], dtype=tf.float64))

y = tf.nn.softmax(tf.matmul(x, w) + b)
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float64"))

ti, tl, vi, vl = dataset.GetDataset(0.2, True, True)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
batch_size = 32
epoch = 10
print sess.run(accuracy, feed_dict={x: vi, y_: vl})
for e in range(epoch):
    for i in range(len(ti) / batch_size):
        batch_start = i * batch_size
        batch_end = batch_start + batch_size
        batch_images = ti[batch_start:batch_end]
        batch_labels = tl[batch_start:batch_end]
        sess.run(train_step, feed_dict={x: batch_images, y_: batch_labels})
    print sess.run(accuracy, feed_dict={x: vi, y_: vl})