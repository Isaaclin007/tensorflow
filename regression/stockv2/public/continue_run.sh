#!/bin/bash

# rm -rf ckpt
# python train.py split_random
python train.py
tensorboard --logdir=./ckpt

