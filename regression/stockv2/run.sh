#!/bin/bash

cd public
./download_model.sh
./cp_model.sh
cd ..
python dqn_fix.py -mode dqntestall
