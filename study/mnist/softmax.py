import tensorflow as tf
import numpy as np
import os
import dataset

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

x = tf.placeholder("float32", shape=[None, 784])
y_ = tf.placeholder("float32", shape=[None, 10])

w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, w) + b)
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float32"))

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