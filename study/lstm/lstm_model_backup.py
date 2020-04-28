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
import dataset

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# dataset_type = 'mnist'
dataset_type = 'seq'

class LstmModel():
    def __init__(self,
                 batch_size,
                 num_steps,
                 input_vec_size,
                 num_classes,
                 lstm_size,
                 num_layers,
                 is_training=True,
                 learning_rate=0.001,
                 grad_clip=5):
        print("--tensorflow version:", tf.__version__)
        print("--tensorflow path:", tf.__path__)
        #batch的大小和截断长度
        self.batch_size = batch_size
        #等同其他地方的time_steps
        self.num_steps = num_steps
        #词向量大小(等同其他地方的input_size)   embedding_size
        self.input_vec_size = input_vec_size
        #输出的类型数（词数）
        self.num_classes = num_classes
        #LSTM隐藏层神经元数:num_units，hidden_size
        self.lstm_size = lstm_size
        #LSTM隐藏层层数
        self.num_layers = num_layers
        #是否是训练状态
        self.is_training = is_training
        #学习率
        self.learning_rate = learning_rate
        #梯度裁剪
        self.grad_clip = grad_clip

    def build_inputs(self,batch_size, num_steps, input_vec_size, num_classes):
        # 输入定义数据占位符(TensorFlow默认使用GPU可能导致参数更新过慢,所以建议参考项目中的代码，尤其在定义Variables时注意要绑定CPU)
        with tf.device("/cpu:0"):
            # 输入的词矩阵,维度为batch_size * num_steps * input_vec_size
            inputs = tf.placeholder(tf.float32, shape=(batch_size, num_steps, input_vec_size), name='inputs')
            #预期输出 batch_size * num_classes
            labels = tf.placeholder(tf.float32, shape=(batch_size, num_classes), name='labels')
            #节点不被dropout的概率
            keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        return inputs,labels,keep_prob

    def build_input_layer(self,input_data, num_steps, input_vec_size, lstm_size):
        with tf.variable_scope("input_wb"):
            with tf.device("/cpu:0"):
                input_wight = tf.Variable(tf.truncated_normal([input_vec_size, lstm_size]))
                input_bias = tf.Variable(tf.zeros([lstm_size, ]))
        tf.summary.histogram("input_weight",input_wight)
        tf.summary.histogram("input_bias", input_bias)
        #首先将向量转换为矩阵
        inputs_data = tf.reshape(input_data, shape=[-1, input_vec_size])
        #执行运算
        rnn_inputs = tf.matmul(inputs_data, input_wight) + input_bias
        #add 将输入运用sigmoid激活函数
        #rnn_inputs = tf.nn.sigmoid(rnn_inputs)
        #将数据再转换为隐藏层需要的格式 [batch_size,num_steps,lstm_size]
        self.rnn_inputs = tf.reshape(rnn_inputs, shape=[-1, num_steps, lstm_size])
        return self.rnn_inputs

    def build_lstm_layer(self,lstm_size, num_layers, batch_size, keep_prob):
        lstm_cell = tf.nn.rnn_cell.LSTMCell(lstm_size, state_is_tuple=True)
        with tf.name_scope('dropout'):
            if self.is_training:
                # 添加dropout.为了防止过拟合，在它的隐层添加了 dropout 正则
                lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)
                tf.summary.scalar('dropout_keep_probability', keep_prob)
        #堆叠多个LSTM单元
        stacked_lstm = tf.nn.rnn_cell.MultiRNNCell([lstm_cell for _ in range(num_layers)], state_is_tuple=True)
        #初始化 LSTM 存储状态.batch_size,stacked_lstm.state_size
        initial_state = stacked_lstm.zero_state(batch_size, tf.float32)
        return stacked_lstm, initial_state

    '''
    构造输出层，与LSTM层 进行全连接
    :param lstm_output  lstm层的输出结果,[batch_size,num_steps,lstm_size]
    :return:
    '''
    def build_output_layer(self,hidden_output, lstm_size, num_classes):
        with tf.variable_scope("softwax"):
            softmax_w = tf.Variable(tf.truncated_normal([lstm_size,num_classes]))
            softmax_b = tf.Variable(tf.zeros(num_classes))
        hidden_output = tf.transpose(hidden_output, [1, 0, 2])
        hidden_output = tf.gather(hidden_output, int(hidden_output.get_shape()[0]) - 1)
        #计算logits
        logits = tf.matmul(hidden_output,softmax_w) + softmax_b
        #输出层softmax返回概率分布
        softmax_out = tf.nn.softmax(logits,name='predictions')
        return softmax_out

    # 注意：下面的方式可以实现“多目标”的场景
    # def build_output_layer(self,hidden_output, lstm_size, num_classes):
    #     with tf.variable_scope("output_wb"):
    #         with tf.device("/cpu:0"):
    #             output_w = tf.Variable(tf.truncated_normal([lstm_size, num_classes]))
    #             output_b = tf.Variable(tf.zeros(num_classes))
    #     tf.summary.histogram("output_weight", output_w)
    #     tf.summary.histogram("output_bias", output_b)
    #         # 将输出的维度进行转换(B,T,D) => (T,B,D)
    #     hidden_output = tf.transpose(hidden_output, [1, 0, 2])
    #     #这里取最后个num_steps得到的数据
    #     hidden_output = tf.gather(hidden_output, int(hidden_output.get_shape()[0]) - 1)
    #     #计算并得出结果
    #     output_vec = tf.matmul(hidden_output, output_w) + output_b
    #     #预测结果
    #     product = tf.nn.sigmoid(output_vec)
    #     return output_vec, product

    def build_loss(self,output_vec, labels):
        #根据logits和labels计算损失。
        #logits：[batch_size,num_classes]; labels：[batch_size,num_classes]
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_vec, labels=labels))
        self.loss_summary = tf.summary.scalar('loss', loss)
        return loss

    # 注意：下面的方式可以实现“多目标”的场景
    # def build_loss(self,output_vec, labels):
    #     # 根据output_vec和labels计算损失。
    #     #output_vec 未经过sigmod或softmax处理的输出
    #     #logits：[batch_size,num_classes]; labels：[batch_size,num_classes]
    #     loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    #         logits=output_vec, labels=labels))
    #     return loss

    def build_optimizer(self,loss, learning_rate, grad_clip):
        # 构造加速训练的优化方法
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
            train_op = optimizer.minimize(loss)
        return train_op

    '''
    定义计算模型预测结果准确度
    '''
    def accuracy_eval(self, predict, y):
        correct_pred = tf.equal(tf.argmax(predict, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        return accuracy

    def build_model(self):
        self.inputs, self.labels, self.keep_prob = self.build_inputs(self.batch_size, self.num_steps, self.input_vec_size, self.num_classes)
        #输入层
        rnn_inputs = self.build_input_layer(self.inputs, self.num_steps, self.input_vec_size, self.lstm_size)
        #隐藏层
        stacked_lstm, self.initial_state = self.build_lstm_layer(self.lstm_size, self.num_layers, self.batch_size, self.keep_prob)
        hidden_output, self.final_state = tf.nn.dynamic_rnn(stacked_lstm, rnn_inputs, initial_state=self.initial_state)
        #输出层
        # output_vec, self.predict = self.build_output_layer(hidden_output, self.lstm_size, self.num_classes)
        self.predict = self.build_output_layer(hidden_output, self.lstm_size, self.num_classes)
        if self.is_training :
            # 使用损失函数
            self.loss = self.build_loss(self.predict, self.labels)
            #使用优化器
            self.train_op = self.build_optimizer(self.loss, self.learning_rate, self.grad_clip)

            self.accuracy = self.accuracy_eval(self.predict, self.labels)
        return self.predict



class Training:
    def __init__(self, 
                 batch_size, 
                 num_steps, 
                 input_vec_size, 
                 num_classes, 
                 lstm_size, 
                 num_layers,
                 epoch_size=1000,
                 learning_rate=0.001, 
                 keep_prob=0.75,
                 grad_clip=5, 
                 checkpoint_dir='./checkpoints',
                 log_dir='./logs'):

        self.epoch_size = epoch_size
        self.keep_prob = keep_prob
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.global_step = tf.Variable(0,name='global_step',trainable=False)
        self.model = LstmModel(batch_size=batch_size, 
                               num_steps=num_steps, 
                               input_vec_size=input_vec_size,
                               num_classes=num_classes, 
                               lstm_size=lstm_size, 
                               num_layers=num_layers,
                               learning_rate=learning_rate,
                               is_training=True,
                               grad_clip=grad_clip)
        self.model.build_model()
        self.init_or_load_session()

    def train(self):
        self.summary_writer = tf.summary.FileWriter(self.log_dir,tf.get_default_graph())
        self.summary_log = tf.summary.merge_all()
        acc_summary = tf.summary.scalar('acc', self.model.accuracy)
        self.current_epoch = 0
        self.counter = 0
        if dataset_type == 'mnist':
            data_sets = dataset.MnistDataset(0.07, 0.14, False, True)
        elif dataset_type == 'seq':
            data_sets = dataset.TestSeqDataset(600000, self.model.num_steps, 0.07, 0.14, False, True)
        for epoch in range(self.epoch_size):
            print('epoch:{}'.format(epoch))
            self.current_epoch = epoch
            state = self.sess.run(self.model.initial_state)
            data_sets.train.reset()
            while (1):
                self.counter += 1
                x, y = data_sets.train.next_batch(self.model.batch_size)
                if len(x) == 0:
                    break
                state, pridect = self.optimization(x,y,state)
            # data_sets.test.current_pos = 0
            # tx, ty = data_sets.test.next_batch(self.model.batch_size)
            # feed_dict = {self.model.inputs: tx,
            #                 self.model.labels: ty,
            #                 self.model.keep_prob:1.}
            # summary_log = self.sess.run(acc_summary,feed_dict=feed_dict)
            # self.summary_writer.add_summary(summary_log, epoch)
            self.evalute(data_sets.test)
            self.evaluation()
        print('training end')

    def evalute(self, test_dataset):
        test_dataset.reset()
        acc_list = []
        while True:
            tx, ty = test_dataset.next_batch(self.model.batch_size)
            if len(tx) != self.model.batch_size:
                break
            feed_dict = {self.model.inputs: tx,
                            self.model.labels: ty,
                            self.model.keep_prob:1.}
            acc = self.sess.run(self.model.accuracy,feed_dict=feed_dict)
            acc_list.append(acc)
        avg_acc = np.mean(acc_list)
        print('test acc：%.3f' % avg_acc)

    '''
        初始化或加载Session
    '''
    def init_or_load_session(self):
        self.sess = tf.Session()
        self.saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(checkpoint_dir=self.checkpoint_dir)
        if ckpt:
            print('restore session from ',ckpt)
            self.saver.restore(self.sess, ckpt)
        else:
            print('initialize all variables')
            self.sess.run(tf.initialize_all_variables())

    def evaluation(self):
        self.saver.save(self.sess,
                        self.checkpoint_dir+'/model{}.ckpt'.format(self.current_epoch),
                        global_step=self.global_step)

    def optimization(self,batch_x,batch_y,state):
        feed_dict = {self.model.inputs: batch_x,
                     self.model.labels: batch_y,
                     self.model.keep_prob: self.keep_prob,
                     self.model.initial_state:state}
        final_state, train_op, batch_loss, pridect,summary_log = self.sess.run([self.model.final_state,
                                                           self.model.train_op,
                                                           self.model.loss,
                                                           self.model.predict,
                                                           self.summary_log],
                                                          feed_dict=feed_dict)
        if self.counter % 1 == 0:
            self.summary_writer.add_summary(summary_log, self.counter)
        #     print('loss: {:.4f}... '.format(batch_loss))
        return final_state,pridect

if __name__ == "__main__":
    #训练模型
    if dataset_type == 'mnist':
        batch_size = 100  # 单个batch中序列的个数
        num_steps = 28  # 单个序列中的字符数目
        input_vec_size = 28  # 隐层节点个数,输入神经元数(单词向量的长度)
        num_classes = 10  # 输出神经元数(最后输出的类别总数，例如这的基站数)
        lstm_size = 32
        num_layers = 2  # LSTM层个数

        learning_rate = 0.01  # 学习率
        #feed in 1 when testing, 0.75 when training
        keep_prob = 0.75  # 训练时dropout层中保留节点比例
        epoch_size = 100  # 迭代次数
    elif dataset_type == 'seq':
        batch_size = 1024
        num_steps = 1
        input_vec_size = 2
        num_classes = 2
        lstm_size = 4
        num_layers = 2
        learning_rate = 0.004
        keep_prob = 0.75
        epoch_size = 10


    training = Training(batch_size=batch_size, 
                        num_steps=num_steps, 
                        input_vec_size=input_vec_size,
                        num_classes=num_classes, 
                        lstm_size=lstm_size, 
                        num_layers=num_layers,
                        learning_rate=learning_rate,
                        epoch_size=epoch_size)
    training.train()
    
    


