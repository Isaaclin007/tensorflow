# -*- coding:UTF-8 -*-


import numpy as np
import dataset
import lstm_model

if __name__ == "__main__":
    training = lstm_model.TFLstm(batch_size=100, 
                                    num_steps=28, 
                                    vec_size=28,
                                    num_classes=10, 
                                    lstm_size=8, 
                                    lstm_layers_num=1,
                                    learning_rate=0.01,
                                    keep_prob=0.5,
                                    grad_clip=5, 
                                    checkpoint_dir='./ckpt_mnist',
                                    log_dir='./ckpt_mnist')
    datasets = dataset.MnistDataset(0.0, 0.1, False, True)
    train_x, train_y = datasets.train.NextBatch(-1)
    test_x, test_y = datasets.test.NextBatch(-1)

    training.Fit(train_x, train_y, test_x, test_y, 10)
    
    


