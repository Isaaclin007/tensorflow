#!/bin/bash

for((i=1;i<=10;i++));
do
python download_train_test_data.py
done

python update_train_test_data.py

