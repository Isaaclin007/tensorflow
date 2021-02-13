# -*- coding:UTF-8 -*-


import tensorflow as tf
from tensorflow.contrib import rnn 
# import mnist dataset
from tensorflow.examples.tutorials.mnist importinput_data
mnist =input_data.read_data_sets ("/tmp/data/",one_hot =True )
# define constants
# unrolled through 28 time steps
time_steps =28 
# hidden LSTM units
num_units =128 
#rows of 28 pixel
sn_input =28 
#learning rate for adam
learning_rate =0.001 
#mnist is meant to be classified in 10 classes(0-9).
n_classes =10 
#size of batch
batch_size =128

#weights and biases of appropriate shape to accomplish above taskout_weights=tf.Variable(tf.random_normal([num_units,n_classes]))out_bias=tf.Variable(tf.random_normal([n_classes]))#defining placeholders#input image placeholder x=tf.placeholder("float",[None,time_steps,n_input])#input label placeholdery=tf.placeholder("float",[None,n_classes])