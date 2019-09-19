#!/bin/bash

#rm -rf model
#scp -r videos@10.100.8.74:~/bianjs/gpu_train/model ./
rsync -avzu --progress videos@10.100.8.74:~/bianjs/gpu_train/model ./
