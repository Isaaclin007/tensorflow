#!/bin/bash

# scp ./data/* videos@10.100.8.74:~/bianjs/gpu_train/data/

rsync -avzu --progress ./data/* videos@10.100.8.74:~/bianjs/gpu_train/data/