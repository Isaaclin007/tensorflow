# -*- coding:UTF-8 -*-


import numpy as np
import pandas as pd
import os
import time
import datetime
import sys
import math
import random
import tensorflow as tf
import dataset_common

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# dataset_type = 'mnist'
dataset_type = 'seq'
SET_INITIAL_STATE = False
SUMMARY_TRAIN_LOSS = True

class LstmModel():
    def __init__(self,
                 batch_size,
                 num_steps,
                 vec_size,
                 num_classes,
                 lstm_size,
                 lstm_layers_num,
                 is_training=True,
                 learning_rate=0.001,
                 grad_clip=5):
        # print("--tensorflow version:", tf.__version__)
        # print("--tensorflow path:", tf.__path__)
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.vec_size = vec_size
        self.num_classes = num_classes
        self.lstm_size = lstm_size
        self.lstm_layers_num = lstm_layers_num
        self.is_training = is_training
        self.learning_rate = learning_rate
        self.grad_clip = grad_clip  # 没有使用


    def BuildModel(self):
        batch_size = self.batch_size
        num_steps = self.num_steps
        vec_size = self.vec_size
        num_classes = self.num_classes
        lstm_size = self.lstm_size
        lstm_layers_num = self.lstm_layers_num
        learning_rate = self.learning_rate

        # input placeholder
        with tf.device("/cpu:0"):
            inputs = tf.placeholder(tf.float32, shape=(batch_size, num_steps, vec_size), name='inputs')
            labels = tf.placeholder(tf.float32, shape=(batch_size, num_classes), name='labels')
            keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        # input placeholder -> rnn input
        with tf.variable_scope("input_wb"):
            with tf.device("/cpu:0"):
                input_wight = tf.Variable(tf.truncated_normal([vec_size, lstm_size]))
                input_bias = tf.Variable(tf.zeros([lstm_size, ]))
        inputs_reshape = tf.reshape(inputs, shape=[-1, vec_size])
        rnn_inputs = tf.matmul(inputs_reshape, input_wight) + input_bias
        #rnn_inputs = tf.nn.sigmoid(rnn_inputs)
        rnn_inputs = tf.reshape(rnn_inputs, shape=[-1, num_steps, lstm_size])

        # lstm layer, input -> hidden_output
        lstm_cell = tf.nn.rnn_cell.LSTMCell(lstm_size, state_is_tuple=True)
        with tf.name_scope('dropout'):
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)
        stacked_lstm = tf.nn.rnn_cell.MultiRNNCell([lstm_cell for _ in range(lstm_layers_num)], state_is_tuple=True)
        initial_state = stacked_lstm.zero_state(batch_size, tf.float32)
        if SET_INITIAL_STATE:
            hidden_output, final_state = tf.nn.dynamic_rnn(stacked_lstm, rnn_inputs, initial_state=initial_state)
        else:
            hidden_output, final_state = tf.nn.dynamic_rnn(stacked_lstm, rnn_inputs, dtype=tf.float32)

        # hidden_output -> softmax_out
        with tf.variable_scope("softwax"):
            softmax_w = tf.Variable(tf.truncated_normal([lstm_size, num_classes]))
            softmax_b = tf.Variable(tf.zeros(num_classes))
        hidden_output = tf.transpose(hidden_output, [1, 0, 2])
        hidden_output = tf.gather(hidden_output, int(hidden_output.get_shape()[0]) - 1)
        logits = tf.matmul(hidden_output,softmax_w) + softmax_b
        predict = tf.nn.softmax(logits, name='predictions')

        # acc
        correct_pred = tf.equal(tf.argmax(predict, 1), tf.argmax(labels, 1))
        acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        
        # loss
        # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict, labels=labels))
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=predict, labels=labels))

        global_step = tf.Variable(0, name='global_step',trainable=False)
        epoch = tf.Variable(0, name='epoch',trainable=False)
        epoch_add = tf.assign(epoch, tf.add(epoch, tf.constant(1)))
        
        # Optimizer
        with tf.name_scope('train'):
            optimizer = tf.train.AdamOptimizer(learning_rate)
            '''
            optimizer = tf.train.AdadeltaOptimizer(learning_rate)
            optimizer = tf.train.AdagradOptimizer(learning_rate)
            optimizer = tf.train.FtrlOptimizer(learning_rate)
            optimizer = tf.train.RMSPropOptimizer(learning_rate)
            '''
            #optimizer.apply_gradients(zip(grads,tvar))
            #该函数是简单的合并了compute_gradients()与apply_gradients()函数，返回为一个优化更新后的var_list
            #如果global_step非None，该操作还会为global_step做自增操作
            train = optimizer.minimize(loss, global_step=global_step)
        self.inputs = inputs
        self.labels = labels
        self.keep_prob = keep_prob
        self.initial_state = initial_state
        self.predict = predict
        self.train = train
        self.global_step = global_step
        self.epoch = epoch
        self.epoch_add = epoch_add
        self.acc = acc
        self.loss = loss
        self.step_acc_summary = tf.summary.scalar('step_acc', acc)
        self.step_loss_summary = tf.summary.scalar('step_loss', loss)
        self.epoch_acc_summary = tf.summary.scalar('epoch_acc', acc)
        self.epoch_loss_summary = tf.summary.scalar('epoch_loss', loss)

        self.step_summary = tf.summary.merge([self.step_acc_summary, self.step_loss_summary])
        self.epoch_summary = tf.summary.merge([self.epoch_acc_summary, self.epoch_loss_summary])
        self.initial_state_value = None

        # input placeholder
        with tf.device("/cpu:0"):
            p_inputs = tf.placeholder(tf.float32, shape=(None, num_steps, vec_size), name='inputs')
            p_labels = tf.placeholder(tf.float32, shape=(None, num_classes), name='labels')
        p_inputs_reshape = tf.reshape(p_inputs, shape=[-1, vec_size])
        p_rnn_inputs = tf.matmul(p_inputs_reshape, input_wight) + input_bias
        # p_rnn_inputs = tf.nn.sigmoid(p_rnn_inputs)
        p_rnn_inputs = tf.reshape(p_rnn_inputs, shape=[-1, num_steps, lstm_size])

        p_hidden_output, p_final_state = tf.nn.dynamic_rnn(stacked_lstm, p_rnn_inputs, dtype=tf.float32)

        p_hidden_output = tf.transpose(p_hidden_output, [1, 0, 2])
        p_hidden_output = tf.gather(p_hidden_output, int(p_hidden_output.get_shape()[0]) - 1)
        p_logits = tf.matmul(p_hidden_output,softmax_w) + softmax_b
        p_predict = tf.nn.softmax(p_logits, name='predictions')

        # acc
        p_correct_pred = tf.equal(tf.argmax(p_predict, 1), tf.argmax(p_labels, 1))
        p_acc = tf.reduce_mean(tf.cast(p_correct_pred, tf.float32))
        
        # loss
        # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict, labels=labels))
        p_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=p_predict, labels=p_labels))

        self.p_inputs = p_inputs
        self.p_labels = p_labels
        self.p_predict = p_predict
        self.p_acc = p_acc
        self.p_loss = p_loss
        self.p_epoch_acc_summary = tf.summary.scalar('epoch_acc', p_acc)
        self.p_epoch_loss_summary = tf.summary.scalar('epoch_loss', p_loss)
        self.p_epoch_summary = tf.summary.merge([self.p_epoch_acc_summary, self.p_epoch_loss_summary])

    def SetInitialStateValue(self, sess, feed_dict):
        if self.initial_state_value == None:
            self.initial_state_value = sess.run(self.initial_state)
        feed_dict[self.initial_state] = self.initial_state_value

    def Fit(self, sess, batch_x, batch_y, keep_prob):
        feed_dict = {self.inputs: batch_x,
                     self.labels: batch_y,
                     self.keep_prob: keep_prob}
        if SET_INITIAL_STATE:
            self.set_initial_state_value(sess, feed_dict)
        if SUMMARY_TRAIN_LOSS:
            _, summary_log = sess.run([self.train,
                                        self.step_summary], 
                                        feed_dict=feed_dict)
            return summary_log
        else:
            _ = sess.run([self.train], feed_dict=feed_dict)
            return None

    def Evaluate(self, sess, x, y):
        feed_dict = {self.p_inputs: x,
                     self.p_labels: y,
                     self.keep_prob: 1.}
        summary_log = sess.run(self.p_epoch_summary,feed_dict=feed_dict)
        return summary_log

    def Acc(self, sess, x, y):
        feed_dict = {self.p_inputs: x,
                     self.p_labels: y,
                     self.keep_prob: 1.}
        return sess.run(self.p_acc, feed_dict=feed_dict)

    def Loss(self, sess, x, y):
        feed_dict = {self.inputs: x,
                     self.labels: y,
                     self.keep_prob: 1.}
        return sess.run(self.p_loss, feed_dict=feed_dict)

    def Epoch(self, sess):
        return sess.run(self.epoch)

    def EpochAdd(self, sess):
        return sess.run(self.epoch_add)

def RmDir(path):
    if os.path.exists(path):
        ls = os.listdir(path)
        for i in ls:
            c_path = os.path.join(path, i)
            if os.path.isdir(c_path):
                RmDir(c_path)
            else:
                os.remove(c_path)
        os.rmdir(path)

class TFLstm:
    def __init__(self, 
                 batch_size, 
                 num_steps, 
                 vec_size, 
                 num_classes, 
                 lstm_size, 
                 lstm_layers_num,
                 learning_rate=0.001, 
                 keep_prob=0.75,
                 grad_clip=5, 
                 checkpoint_dir='./checkpoints',
                 log_dir='./logs',
                 continue_train=False):

        self.keep_prob = keep_prob
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        self.continue_train = continue_train
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.model = LstmModel(batch_size=batch_size, 
                               num_steps=num_steps, 
                               vec_size=vec_size,
                               num_classes=num_classes, 
                               lstm_size=lstm_size, 
                               lstm_layers_num=lstm_layers_num,
                               learning_rate=learning_rate,
                               is_training=True,
                               grad_clip=grad_clip)
        self.model.BuildModel()
        self.InitOrLoadSession()

    def Fit(self, train_x, train_y, test_x, test_y, epoch):
        if not self.continue_train:
            self.Clean()
        start_time = time.time()
        dataset = dataset_common.DatasetUnit()
        dataset.Init(train_x, train_y)
        self.summary_writer_train = tf.summary.FileWriter(self.log_dir + '/train', self.sess.graph)
        self.summary_writer_test = tf.summary.FileWriter(self.log_dir + '/test', self.sess.graph)
        self.counter = self.global_step_value
        for i in range(epoch):
            dataset.ResetPos()
            while (1):
                x, y = dataset.NextBatch(self.model.batch_size)
                if len(x) == 0:
                    break
                summary_log = self.model.Fit(self.sess, x, y, self.keep_prob)
                if SUMMARY_TRAIN_LOSS:
                    self.summary_writer_train.add_summary(summary_log, self.counter)
                self.counter += 1

            self.epoch_value = self.model.EpochAdd(self.sess)
            self.EvaluteSummary(train_x, train_y, test_x, test_y)
            self.saver.save(self.sess,
                            self.checkpoint_dir + ('/model.ckpt-%u' % self.epoch_value),
                            global_step = self.model.global_step)
            
            sys.stdout.write('\r%d' % (self.epoch_value))
            sys.stdout.flush()
        end_time = time.time()
        print("\ntrain time: %.2f" % (end_time - start_time))

    def Clean(self):
        print('Clean')
        RmDir(self.log_dir + '/train')
        RmDir(self.log_dir + '/test')
        self.sess.run(tf.global_variables_initializer())
        self.epoch_value = self.sess.run(self.model.epoch)
        self.global_step_value = self.sess.run(self.model.global_step)

    def InitOrLoadSession(self):
        self.sess = tf.Session()
        self.saver = tf.train.Saver(max_to_keep=5)
        ckpt = tf.train.latest_checkpoint(checkpoint_dir=self.checkpoint_dir)
        if ckpt:
            print('restore session from: {}'.format(ckpt))
            self.saver.restore(self.sess, ckpt)
        else:
            print('initialize all variables')
            self.sess.run(tf.global_variables_initializer())
        self.epoch_value = self.sess.run(self.model.epoch)
        self.global_step_value = self.sess.run(self.model.global_step)
        print('epoch: {}, global_step: {}'.format(self.epoch_value, self.global_step_value))

    def EvaluteSummary(self, train_x, train_y, test_x, test_y):
        summary_log = self.model.Evaluate(self.sess, train_x, train_y)
        self.summary_writer_train.add_summary(summary_log, self.epoch_value)

        summary_log = self.model.Evaluate(self.sess, test_x, test_y)
        self.summary_writer_test.add_summary(summary_log, self.epoch_value)





    
    


