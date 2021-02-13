#!/bin/bash

python mnist_with_summaries.py --fake_data=true --log_dir=./log
tensorboard --logdir='./log'



