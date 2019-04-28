#!/bin/bash

for((i=1;i<=100;i++));
do
python download_train_test_data.py
sleep 1s
done

python update_preprocess_data.py
python update_train_test_data.py

