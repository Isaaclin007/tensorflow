#!/bin/bash

rm -r ./checkpoints
rm -r ./logs
python lstm_model.py
# tensorboard --logdir=./logs

