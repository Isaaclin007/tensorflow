#!/bin/bash

rm -f *.pyc
cp ./dqn_fix* ../gpu_train/
cp ./dqn_test.py ../gpu_train/
cp ./loss.py ../gpu_train/
cp ./dqn_fix_run.sh ../gpu_train/
rm -f ../gpu_train/*.pyc
scp ../gpu_train/* videos@10.100.8.74:~/bianjs/gpu_train/

