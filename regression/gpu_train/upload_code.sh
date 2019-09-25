#!/bin/bash

rm -f *.pyc
scp -r ./*gpu_train* videos@10.100.8.74:~/bianjs/gpu_train/

