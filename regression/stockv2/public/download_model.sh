#!/bin/bash

# rm -rf model
#scp -r videos@10.100.8.74:~/bianjs/gpu_train/model ./
rsync -avzu --progress videos@10.100.8.74:~/bianjs/gpu_train/model ./

rm ../model/0/*
cp model/0/* ../model/0/
