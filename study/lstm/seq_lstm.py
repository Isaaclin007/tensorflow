# -*- coding:UTF-8 -*-


import numpy as np
import dataset
import lstm_model   

if __name__ == "__main__":
    training = lstm_model.Training(batch_size=1024, 
                                    num_steps=10, 
                                    vec_size=2,
                                    num_classes=2, 
                                    lstm_size=8, 
                                    lstm_layers_num=1,
                                    learning_rate=0.01,
                                    keep_prob=0.5,
                                    grad_clip=5, 
                                    checkpoint_dir='./ckpt_seq',
                                    log_dir='./ckpt_seq')
    datasets = dataset.TestSeqDataset(60000, 10, 0.0, 0.1, False, True)
    train_x, train_y = datasets.train.NextBatch(-1)
    test_x, test_y = datasets.test.NextBatch(-1)

    training.Fit(train_x, train_y, test_x, test_y, 10)
    
    


