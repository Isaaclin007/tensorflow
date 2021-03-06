#!/bin/bash

rm -f *.pyc
# python gpu_train.py fix TP0MaxRatio KerasRMSProp 1 72
# python gpu_train.py fix TP0MaxRatio KerasRMSProp 1 64
# python gpu_train.py fix TP0MaxRatio KerasRMSProp 1 60
# python gpu_train.py fix TP0MaxRatio KerasRMSProp 1 56
# python gpu_train.py fix TP0MaxRatio KerasRMSProp 1 48
# python gpu_train.py fix TP0MaxRatio KerasRMSProp 1 36
# python gpu_train.py fix TP0MaxRatio KerasRMSProp 1 32
# python gpu_train.py fix TP0MaxRatio KerasRMSProp 1 16
# python gpu_train.py fix TP0MaxRatio KerasRMSProp 1 8
# python gpu_train.py fix TP0MaxRatio KerasRMSProp 1 4

# python gpu_train.py fix TP0MaxRatio KerasRMSProp 2 64
# python gpu_train.py fix TP0MaxRatio KerasRMSProp 3 64
# python gpu_train.py fix TP0MaxRatio KerasRMSProp 4 64 
# python gpu_train.py fix TP0MaxRatio KerasRMSProp 5 64
# python gpu_train.py fix TP0MaxRatio KerasRMSProp 6 64
# python gpu_train.py fix TP0MaxRatio KerasRMSProp 7 64
# python gpu_train.py fix TP0MaxRatio KerasRMSProp 8 64
# python gpu_train.py fix TP0MaxRatio KerasRMSProp 9 64

# python gpu_train.py fix Ts5Ps50MaxRatio KerasRMSProp 1 64
# python gpu_train.py fix Ts5Ps50MaxRatio KerasRMSProp 1 36

# python gpu_train.py wave TP0MaxRatio KerasRMSProp 1 64
# python gpu_train.py wave TP0MaxRatio KerasRMSProp 1 48
# python gpu_train.py wave TP0MaxRatio KerasRMSProp 1 36
# python gpu_train.py wave TP0MaxRatio KerasRMSProp 1 32
# python gpu_train.py wave TP0MaxRatio KerasRMSProp 1 24
# python gpu_train.py wave TP0MaxRatio KerasRMSProp 1 16

# python gpu_train.py wave Ts5Ps50MaxRatio KerasRMSProp 1 64
# python gpu_train.py wave Ts5Ps50MaxRatio KerasRMSProp 1 36

# python gpu_train.py wave Abs KerasRMSProp 1 36

# python gpu_train.py wave Ts9Ps90MaxRatio KerasRMSProp 1 36

# python gpu_train.py wave TP0MaxRatio KerasRMSProp 1 36 0.001
# python gpu_train.py wave TP0MaxRatio KerasRMSProp 1 36 0.002
# python gpu_train.py wave TP0MaxRatio KerasRMSProp 1 36 0.008
# python gpu_train.py wave TP0MaxRatio KerasRMSProp 1 36 0.016
# python gpu_train.py wave TP0MaxRatio KerasRMSProp 1 36 0.032
# python gpu_train.py wave TP0MaxRatio KerasRMSProp 1 36 0.064

# python gpu_train.py wave TP0MaxRatio KerasRMSProp 1 36 0.004 20480
# python gpu_train.py wave TP0MaxRatio KerasRMSProp 1 36 0.004 40960
# python gpu_train.py wave TP0MaxRatio KerasRMSProp 1 36 0.004 81920

# python gpu_train.py wave mae KerasRMSProp 1 36 0.004 81920
# python gpu_train.py wave mae KerasRMSProp 1 8 0.004 81920

# python gpu_train.py wave TP0MaxRatio KerasRMSProp 1 36 0.004 10240
# python gpu_train.py wave TP0MaxRatio KerasRMSProp 1 35 0.004 10240
# python gpu_train.py wave TP0MaxRatio KerasRMSProp 1 37 0.004 10240

# python gpu_train.py wave TP0MaxRatio KerasRMSProp 1 64 0.004 10240

# python gpu_train.py wave Ts5Ps50MaxRatio KerasRMSProp 1 64 0.004 10240
# python gpu_train.py wave Ts9Ps90MaxRatio KerasRMSProp 1 64 0.004 10240

# python gpu_train.py wave Ts5Ps50MaxRatio KerasRMSProp 1 64 0.004 10240 500

# python gpu_train.py --train_data=fix --loss_func=LossTanhDiff --model_type=LSTM --lstm_size=64 --learning_rate=0.004 

# python gpu_train.py --train_data=wave --loss_func=LossTanhDiff --model_type=LSTM --lstm_size=64 --learning_rate=0.004 

# python gpu_train.py --train_data=wave --loss_func=LossTanhDiff --model_type=Dense --dense_size=4 --learning_rate=0.004 
# python gpu_train.py --train_data=fix  --loss_func=LossTanhDiff --model_type=Dense --dense_size=4 --learning_rate=0.004 --epochs=500
# python gpu_train.py --train_data=fix  --loss_func=LossTanhDiff --model_type=Dense --dense_size=8 --learning_rate=0.004 --epochs=500
# python gpu_train.py --train_data=fix  --loss_func=LossTanhDiff --model_type=Dense --dense_size=24 --learning_rate=0.004 --epochs=500
# python gpu_train.py --train_data=fix  --loss_func=LossTanhDiff --model_type=Dense --dense_size=32 --learning_rate=0.004 --epochs=500
# python gpu_train.py --train_data=fix  --loss_func=LossTanhDiff --model_type=Dense --dense_size=64 --learning_rate=0.004 --epochs=500
# python gpu_train.py --train_data=fix  --loss_func=LossTanhDiff --model_type=Dense --dense_size=64,32,8 --learning_rate=0.004 --epochs=500

# python gpu_train.py --train_data=wave --loss_func=LossTanhDiff --model_type=LSTM --lstm_size=16 --learning_rate=0.004 --epochs=500
# python gpu_train.py --train_data=wave --loss_func=LossTanhDiff --model_type=LSTM --lstm_size=24 --learning_rate=0.004 --epochs=500
# python gpu_train.py --train_data=wave --loss_func=LossTanhDiff --model_type=LSTM --lstm_size=32 --learning_rate=0.004 --epochs=500
# python gpu_train.py --train_data=wave --loss_func=LossTanhDiff --model_type=LSTM --lstm_size=48 --learning_rate=0.004 --epochs=500
# python gpu_train.py --train_data=wave --loss_func=LossTanhDiff --model_type=LSTM --lstm_size=64 --learning_rate=0.004 --epochs=500
# python gpu_train.py --train_data=wave --loss_func=LossTanhDiff --model_type=LSTM --lstm_size=128 --learning_rate=0.004 --epochs=500
# python gpu_train.py --train_data=wave  --loss_func=LossTanhDiff --model_type=Dense --dense_size=4 --learning_rate=0.004 --epochs=500
# python gpu_train.py --train_data=wave  --loss_func=LossTanhDiff --model_type=Dense --dense_size=8 --learning_rate=0.004 --epochs=500
# python gpu_train.py --train_data=wave  --loss_func=LossTanhDiff --model_type=Dense --dense_size=24 --learning_rate=0.004 --epochs=500
# python gpu_train.py --train_data=wave  --loss_func=LossTanhDiff --model_type=Dense --dense_size=32 --learning_rate=0.004 --epochs=500
# python gpu_train.py --train_data=wave  --loss_func=LossTanhDiff --model_type=Dense --dense_size=64 --learning_rate=0.004 --epochs=500
# python gpu_train.py --train_data=wave  --loss_func=LossTanhDiff --model_type=Dense --dense_size=64,32,8 --learning_rate=0.004 --epochs=500


# python gpu_train.py --train_data=fix --loss_func=LossTanhDiff --model_type=LSTM --lstm_size=16 --learning_rate=0.004 --epochs=500
# python gpu_train.py --train_data=fix --loss_func=LossTanhDiff --model_type=LSTM --lstm_size=24 --learning_rate=0.004 --epochs=500
# python gpu_train.py --train_data=fix --loss_func=LossTanhDiff --model_type=LSTM --lstm_size=32 --learning_rate=0.004 --epochs=500
# python gpu_train.py --train_data=fix --loss_func=LossTanhDiff --model_type=LSTM --lstm_size=48 --learning_rate=0.004 --epochs=500
# python gpu_train.py --train_data=fix --loss_func=LossTanhDiff --model_type=LSTM --lstm_size=64 --learning_rate=0.004 --epochs=500
# python gpu_train.py --train_data=fix --loss_func=LossTanhDiff --model_type=LSTM --lstm_size=128 --learning_rate=0.004 --epochs=500
# python gpu_train.py --train_data=fix  --loss_func=LossTanhDiff --model_type=Dense --dense_size=4 --learning_rate=0.004 --epochs=500
# python gpu_train.py --train_data=fix  --loss_func=LossTanhDiff --model_type=Dense --dense_size=8 --learning_rate=0.004 --epochs=500
# python gpu_train.py --train_data=fix  --loss_func=LossTanhDiff --model_type=Dense --dense_size=24 --learning_rate=0.004 --epochs=500
# python gpu_train.py --train_data=fix  --loss_func=LossTanhDiff --model_type=Dense --dense_size=32 --learning_rate=0.004 --epochs=500
# python gpu_train.py --train_data=fix  --loss_func=LossTanhDiff --model_type=Dense --dense_size=64 --learning_rate=0.004 --epochs=500
# python gpu_train.py --train_data=fix  --loss_func=LossTanhDiff --model_type=Dense --dense_size=64,32,8 --learning_rate=0.004 --epochs=500

# python gpu_train.py --train_data=wave --loss_func=LossTanhDiff --model_type=LSTM --lstm_size=32 --lstm_dense_size=8 --learning_rate=0.004 --epochs=500
# python gpu_train.py --train_data=wave --data_split_mode=random --loss_func=LossTanhDiff --model_type=LSTM --lstm_size=32 --lstm_dense_size=8 --learning_rate=0.004 --epochs=500
# python gpu_train.py --train_data=wave --data_split_mode=samplebydate --loss_func=LossTanhDiff --model_type=LSTM --lstm_size=32 --lstm_dense_size=8 --learning_rate=0.004 --epochs=500
# python gpu_train.py --train_data=wave --loss_func=LossTanhDiff --model_type=LSTM --lstm_size=32 --lstm_dense_size=4 --learning_rate=0.008 --epochs=500

python gpu_train.py --train_data=wave --loss_func=LossTP0MaxRatio --model_type=LSTM --lstm_size=36 --lstm_dense_size=1 --learning_rate=0.004 --epochs=1000 --val_split=0.1



