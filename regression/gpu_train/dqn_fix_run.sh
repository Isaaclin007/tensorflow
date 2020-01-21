#!/bin/bash

python dqn_fix.py -mode train -epoch 1000 -step 10
python dqn_fix.py -mode testall -step 10
python dqn_fix.py -mode show
