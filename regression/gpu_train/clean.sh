#!/bin/bash

ssh videos@10.100.8.74 <<remotessh
cd ~/bianjs/gpu_train
rm *.py*
rm *.sh
rm -r model
remotessh

