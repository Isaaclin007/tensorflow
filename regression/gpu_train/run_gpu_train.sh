#!/bin/bash

rm -f *.pyc
python gpu_train.py fix TP0MaxRatio KerasRMSProp 1 72
python gpu_train.py fix TP0MaxRatio KerasRMSProp 1 64
python gpu_train.py fix TP0MaxRatio KerasRMSProp 1 60
python gpu_train.py fix TP0MaxRatio KerasRMSProp 1 56
python gpu_train.py fix TP0MaxRatio KerasRMSProp 1 48
python gpu_train.py fix TP0MaxRatio KerasRMSProp 1 36
python gpu_train.py fix TP0MaxRatio KerasRMSProp 1 32
python gpu_train.py fix TP0MaxRatio KerasRMSProp 1 16
python gpu_train.py fix TP0MaxRatio KerasRMSProp 1 8
python gpu_train.py fix TP0MaxRatio KerasRMSProp 1 4

python gpu_train.py fix TP0MaxRatio KerasRMSProp 2 64
python gpu_train.py fix TP0MaxRatio KerasRMSProp 3 64
python gpu_train.py fix TP0MaxRatio KerasRMSProp 4 64 
python gpu_train.py fix TP0MaxRatio KerasRMSProp 5 64
python gpu_train.py fix TP0MaxRatio KerasRMSProp 6 64
python gpu_train.py fix TP0MaxRatio KerasRMSProp 7 64
python gpu_train.py fix TP0MaxRatio KerasRMSProp 8 64
python gpu_train.py fix TP0MaxRatio KerasRMSProp 9 64

python gpu_train.py fix Ts5Ps50MaxRatio KerasRMSProp 1 64
python gpu_train.py fix Ts5Ps50MaxRatio KerasRMSProp 1 36

python gpu_train.py wave TP0MaxRatio KerasRMSProp 1 64
python gpu_train.py wave TP0MaxRatio KerasRMSProp 1 48
python gpu_train.py wave TP0MaxRatio KerasRMSProp 1 36
python gpu_train.py wave TP0MaxRatio KerasRMSProp 1 32
python gpu_train.py wave TP0MaxRatio KerasRMSProp 1 24
python gpu_train.py wave TP0MaxRatio KerasRMSProp 1 16

python gpu_train.py wave Ts5Ps50MaxRatio KerasRMSProp 1 64
python gpu_train.py wave Ts5Ps50MaxRatio KerasRMSProp 1 36