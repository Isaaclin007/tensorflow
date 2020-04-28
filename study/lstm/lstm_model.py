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

# def build_lstm_model(batch_size, 
#                     num_steps, 
#                     vec_size, 
#                     num_classes, 
#                     lstm_size, 
#                     lstm_layers_num,  
#                     learning_rate):
#         # input placeholder
#         with tf.device("/cpu:0"):
#             inputs = tf.placeholder(tf.float32, shape=(None, num_steps, vec_size), name='inputs')
#             labels = tf.placeholder(tf.float32, shape=(None, num_classes), name='labels')
#             keep_prob = tf.placeholder(tf.float32, name='keep_prob')

#         # input placeholder -> rnn input
#         with tf.variable_scope("input_wb"):
#             with tf.device("/cpu:0"):
#                 input_wight = tf.Variable(tf.truncated_normal([vec_size, lstm_size]))
#                 input_bias = tf.Variable(tf.zeros([lstm_size, ]))
#         inputs_reshape = tf.reshape(inputs, shape=[-1, vec_size])
#         rnn_inputs = tf.matmul(inputs_reshape, input_wight) + input_bias
#         #rnn_inputs = tf.nn.sigmoid(rnn_inputs)
#         rnn_inputs = tf.reshape(rnn_inputs, shape=[-1, num_steps, lstm_size])

#         # lstm layer, input -> hidden_output
#         lstm_cell = tf.nn.rnn_cell.LSTMCell(lstm_size, state_is_tuple=True)
#         with tf.name_scope('dropout'):
#             lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)
#         stacked_lstm = tf.nn.rnn_cell.MultiRNNCell([lstm_cell for _ in range(lstm_layers_num)], state_is_tuple=True)
#         initial_state = stacked_lstm.zero_state(batch_size, tf.float32)
#         if SET_INITIAL_STATE:
#             hidden_output, final_state = tf.nn.dynamic_rnn(stacked_lstm, rnn_inputs, initial_state=initial_state)
#         else:
#             hidden_output, final_state = tf.nn.dynamic_rnn(stacked_lstm, rnn_inputs, dtype=tf.float32)

#         # hidden_output -> softmax_out
#         with tf.variable_scope("softwax"):
#             softmax_w = tf.Variable(tf.truncated_normal([lstm_size, num_classes]))
#             softmax_b = tf.Variable(tf.zeros(num_classes))
#         hidden_output = tf.transpose(hidden_output, [1, 0, 2])
#         hidden_output = tf.gather(hidden_output, int(hidden_output.get_shape()[0]) - 1)
#         logits = tf.matmul(hidden_output,softmax_w) + softmax_b
#         predict = tf.nn.softmax(logits, name='predictions')

#         # acc
#         correct_pred = tf.equal(tf.argmax(predict, 1), tf.argmax(labels, 1))
#         acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
#         acc_summary = tf.summary.scalar('acc', acc)
        
#         # loss
#         # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict, labels=labels))
#         loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=predict, labels=labels))
#         loss_summary = tf.summary.scalar('loss', loss)

#         global_step = tf.Variable(0, name='global_step',trainable=False)
#         epoch = tf.Variable(0, name='epoch',trainable=False)
#         epoch_add = tf.assign(epoch, tf.add(epoch, tf.constant(1)))
        
#         # Optimizer
#         with tf.name_scope('train'):
#             optimizer = tf.train.AdamOptimizer(learning_rate)
#             '''
#             optimizer = tf.train.AdadeltaOptimizer(learning_rate)
#             optimizer = tf.train.AdagradOptimizer(learning_rate)
#             optimizer = tf.train.FtrlOptimizer(learning_rate)
#             optimizer = tf.train.RMSPropOptimizer(learning_rate)
#             '''
#             #optimizer.apply_gradients(zip(grads,tvar))
#             #该函数是简单的合并了compute_gradients()与apply_gradients()函数，返回为一个优化更新后的var_list
#             #如果global_step非None，该操作还会为global_step做自增操作
#             train = optimizer.minimize(loss, global_step=global_step)
#         return inputs, \
#                labels, \
#                 keep_prob, \
#                 initial_state, \
#                 predict, \
#                 train, \
#                 global_step, \
#                 epoch, \
#                 epoch_add, \
#                 acc, \
#                 acc_summary, \
#                 loss, \
#                 loss_summary

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
            inputs = tf.placeholder(tf.float32, shape=(None, num_steps, vec_size), name='inputs')
            labels = tf.placeholder(tf.float32, shape=(None, num_classes), name='labels')
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
        feed_dict = {self.inputs: x,
                     self.labels: y,
                     self.keep_prob: 1.}
        if SET_INITIAL_STATE:
            self.set_initial_state_value(sess, feed_dict)
        summary_log = sess.run(self.epoch_summary,feed_dict=feed_dict)
        return summary_log

    def Acc(self, sess, batch_x, batch_y):
        feed_dict = {self.inputs: batch_x,
                     self.labels: batch_y,
                     self.keep_prob: 1.}
        if SET_INITIAL_STATE:
            self.set_initial_state_value(sess, feed_dict)
        return sess.run(self.acc_op, feed_dict=feed_dict)

    def Loss(self, sess, batch_x, batch_y):
        feed_dict = {self.inputs: batch_x,
                     self.labels: batch_y,
                     self.keep_prob: 1.}
        if SET_INITIAL_STATE:
            self.set_initial_state_value(sess, feed_dict)
        return sess.run(self.loss_op, feed_dict=feed_dict)

    def Epoch(self, sess):
        return sess.run(self.epoch)

    def EpochAdd(self, sess):
        return sess.run(self.epoch_add)

class Training:
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

    # def train(self):
    #     # self.summary_writer = tf.summary.FileWriter(self.log_dir,tf.get_default_graph())
    #     self.summary_writer_train = tf.summary.FileWriter(self.log_dir + '/train', self.sess.graph)
    #     self.summary_writer_test = tf.summary.FileWriter(self.log_dir + '/test', self.sess.graph)
        
    #     # self.summary_log = tf.summary.merge_all()
    #     self.current_epoch = 0
    #     self.counter = self.global_step_value
    #     if dataset_type == 'mnist':
    #         data_sets = dataset.MnistDataset(0.07, 0.14, False, True)
    #     elif dataset_type == 'seq':
    #         data_sets = dataset.TestSeqDataset(600000, self.model.num_steps, 0.0, 0.1, False, True)
    #     for epoch in range(self.epoch_size):
    #         self.current_epoch = self.model.Epoch(self.sess)
    #         print('epoch:{}'.format(self.current_epoch))
    #         # state = self.sess.run(self.model.initial_state)
    #         data_sets.train.reset()
    #         while (1):
    #             self.counter += 1
    #             x, y = data_sets.train.next_batch(self.model.batch_size)
    #             if len(x) == 0:
    #                 break
    #             summary_log = self.model.fit(self.sess, x, y, self.keep_prob)
    #             # feed_dict = {self.model.inputs: x,
    #             #              self.model.labels: y,
    #             #              self.model.keep_prob: self.keep_prob}
    #             #              #  self.initial_state:state}
    #             # _, summary_log = self.sess.run([self.model.train,
    #             #                                 self.model.train_summary], 
    #             #                                 feed_dict=feed_dict)
    #             if SUMMARY_TRAIN_LOSS:
    #                 self.summary_writer_train.add_summary(summary_log, self.counter)
    #         # data_sets.test.current_pos = 0
    #         # tx, ty = data_sets.test.next_batch(self.model.batch_size)
    #         # summary_log = self.model.evaluate(self.sess, tx, ty)
    #         # summary_writer_test.add_summary(summary_log, self.counter)
    #         self.evalute_summary(data_sets)
    #         self.saver.save(self.sess,
    #                         self.checkpoint_dir + ('/model.ckpt-%u' % self.current_epoch),
    #                         global_step = self.model.global_step)
    #         self.model.EpochAdd(self.sess)
    #         # feed_dict = {self.model.inputs: tx,
    #         #              self.model.labels: ty,
    #         #              self.model.keep_prob: 1.}
    #         # summary_log = self.sess.run(self.model.acc_summary,feed_dict=feed_dict)
    #         # self.summary_writer.add_summary(summary_log, epoch)
    #         # self.evaluation()
    #     print('training end')

    def Fit(self, train_x, train_y, test_x, test_y, epoch):
        start_time = time.time()
        dataset = dataset_common.DatasetUnit()
        dataset.Init(train_x, train_y)
        self.summary_writer_train = tf.summary.FileWriter(self.log_dir + '/train', self.sess.graph)
        self.summary_writer_test = tf.summary.FileWriter(self.log_dir + '/test', self.sess.graph)
        self.counter = self.global_step_value
        for i in range(epoch):
            self.current_epoch = self.model.Epoch(self.sess)
            print('epoch:{}'.format(self.current_epoch))
            # state = self.sess.run(self.model.initial_state)
            dataset.ResetPos()
            while (1):
                x, y = dataset.NextBatch(self.model.batch_size)
                if len(x) == 0:
                    break
                summary_log = self.model.Fit(self.sess, x, y, self.keep_prob)
                if SUMMARY_TRAIN_LOSS:
                    self.summary_writer_train.add_summary(summary_log, self.counter)
                self.counter += 1

            self.EvaluteSummary(train_x, train_y, test_x, test_y)
            self.saver.save(self.sess,
                            self.checkpoint_dir + ('/model.ckpt-%u' % self.current_epoch),
                            global_step = self.model.global_step)
            self.model.EpochAdd(self.sess)
        end_time = time.time()
        print("train time: %.2f" % (end_time - start_time))

    '''
        初始化或加载Session
    '''
    def InitOrLoadSession(self):
        self.sess = tf.Session()
        self.saver = tf.train.Saver(max_to_keep=5)
        ckpt = tf.train.latest_checkpoint(checkpoint_dir=self.checkpoint_dir)
        if ckpt and self.continue_train:
            print('restore session from: {}'.format(ckpt))
            self.saver.restore(self.sess, ckpt)
        else:
            print('initialize all variables')
            self.sess.run(tf.global_variables_initializer())
        self.epoch_value = self.sess.run(self.model.epoch)
        self.global_step_value = self.sess.run(self.model.global_step)
        print('epoch: {}, global_step: {}'.format(self.epoch_value, self.global_step_value))

    def Evalute(self, input_dataset):
        if SET_INITIAL_STATE:
            # test_dataset.reset()
            # acc_list = []
            # while True:
            #     tx, ty = test_dataset.next_batch(self.model.batch_size)
            #     if len(tx) != self.model.batch_size:
            #         break
            #     acc = self.model.acc(self.sess, tx, ty)
            #     acc_list.append(acc)
            # avg_acc = np.mean(acc_list)
            # print('test acc：%.3f' % avg_acc)
            print('not support')
        else:
            x, y = input_dataset.train.next_batch(-1)
            train_acc = self.model.acc(self.sess, x, y)
            x, y = input_dataset.test.next_batch(-1)
            test_acc = self.model.acc(self.sess, x, y)
            print('train / test acc：%.3f / %.3f' % (train_acc, test_acc))

    def EvaluteSummary(self, train_x, train_y, test_x, test_y):
        summary_log = self.model.Evaluate(self.sess, train_x, train_y)
        self.summary_writer_train.add_summary(summary_log, self.current_epoch)

        summary_log = self.model.Evaluate(self.sess, test_x, test_y)
        self.summary_writer_test.add_summary(summary_log, self.current_epoch)

    # def evalute(self, test_dataset):
    #     test_dataset.reset()
    #     tx, ty = test_dataset.next_batch(-1)
    #     print(len(tx))
    #     acc = self.model.acc(self.sess, tx, ty)
    #     print('test acc：%.3f' % acc)

    # def test(self):
    #     if dataset_type == 'mnist':
    #         data_sets = dataset.MnistDataset(0.07, 0.14, False, True)
    #     elif dataset_type == 'seq':
    #         data_sets = dataset.TestSeqDataset(6000000, self.model.num_steps, 0.0, 0.1, False, True)
    #     self.evalute(data_sets.test)



           

if __name__ == "__main__":
    #训练模型
    if dataset_type == 'mnist':
        batch_size = 100  # 单个batch中序列的个数
        num_steps = 28  # 单个序列中的字符数目
        vec_size = 28  # 隐层节点个数,输入神经元数(单词向量的长度)
        num_classes = 10  # 输出神经元数(最后输出的类别总数，例如这的基站数)
        lstm_size = 32
        lstm_layers_num = 2  # LSTM层个数

        learning_rate = 0.01  # 学习率
        #feed in 1 when testing, 0.75 when training
        keep_prob = 0.75  # 训练时dropout层中保留节点比例
        epoch_size = 100  # 迭代次数
    elif dataset_type == 'seq':
        batch_size = 10240
        num_steps = 1
        vec_size = 2
        num_classes = 2
        lstm_size = 4
        lstm_layers_num = 2
        learning_rate = 0.004
        keep_prob = 0.75
        epoch_size = 20


    training = Training(batch_size=batch_size, 
                        num_steps=num_steps, 
                        vec_size=vec_size,
                        num_classes=num_classes, 
                        lstm_size=lstm_size, 
                        lstm_layers_num=lstm_layers_num,
                        learning_rate=learning_rate,
                        epoch_size=epoch_size,
                        grad_clip=5, 
                        checkpoint_dir='./ckpt_mnist',
                        log_dir='./ckpt_mnist')
    datasets = dataset.TestSeqDataset(600000, num_steps, 0.0, 0.1, False, True)
    train_x, train_y = datasets.train.next_batch
    start = time.time()
    training.Fit()
    # training.test()
    end = time.time()
    print("run time: %.2f" % (end - start))
    
    


