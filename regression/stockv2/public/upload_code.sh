#!/bin/bash

cp ../loss.py ./
rm -f *.pyc
scp ./*.py videos@10.100.8.74:~/bianjs/gpu_train/
rm ./loss.py

