# -*- coding:UTF-8 -*-

import numpy
import tensorflow as tf
from tensorflow import keras

# fix random seed for reproducibility
numpy.random.seed(7)
# define the raw dataset

MAX_VALUE = 10

def CreateSeqArray():
    arr = []
    temp0 = 57
    temp1 = 113
    temp2 = 242
    for iloop in range(10000):
        current = (temp0 * 3 + temp1 * 3 + temp2 * 3) % MAX_VALUE
        arr.append(current)
        # arr.append(iloop % MAX_VALUE)
        temp0 = temp1
        temp1 = temp2
        temp2 = current
    # print(arr)
    return arr


alphabet = CreateSeqArray()
seq_length = 1
dataX = []
dataY = []
for i in range(0, len(alphabet) - seq_length, 1):
    seq_in = alphabet[i:i + seq_length]
    seq_out = alphabet[i + seq_length]
    dataX.append(seq_in)
    dataY.append(seq_out)
# convert list of lists to array and pad sequences if needed
X = numpy.array(dataX, dtype=numpy.float)
# reshape X to be [samples, time steps, features]
X = numpy.reshape(dataX, (X.shape[0], seq_length, 1))
# normalize
X = X / float(MAX_VALUE)
# one hot encode the output variable
y = numpy.array(dataY, dtype=numpy.float)
# create and fit the model

keep_stat = False

model = keras.models.Sequential()
if keep_stat:
    model.add(keras.layers.LSTM(16, batch_input_shape=(len(X), X.shape[1], X.shape[2]), stateful=True))
else:
    model.add(keras.layers.LSTM(16, input_shape=(X.shape[1], X.shape[2])))

model.add(keras.layers.Dense(1))
my_optimizer = keras.optimizers.RMSprop(lr=0.01, rho=0.9, epsilon=1e-06)
model.compile(loss='mae', optimizer=my_optimizer, metrics=['mae'])

if keep_stat:
    for iloop in range(5):
        # model.fit(X, y, epochs=1, batch_size=1, verbose=2, shuffle=False)
        model.fit(X, y, epochs=1, batch_size=len(X), verbose=2, shuffle=False)
        model.reset_states()
else:
    model.fit(X, y, epochs=1000, batch_size=len(X), verbose=2, shuffle=True)
# summarize performance of the model
# scores = model.evaluate(X, y, verbose=0)
# print("Model Accuracy: %.2f%%" % (scores[1]*100))
# demonstrate some model predictions
error = 0.0
for iloop in range(len(X)):
    x = X[iloop]
    x = numpy.reshape(x, (1, 1, 1))
    model.reset_states()
    prediction = model.predict(x, verbose=0)
    error += abs(prediction - y[iloop])
    # print seq_in, "->", result
print('error1: %f' % error)

error = 0.0
model.reset_states()
for iloop in range(len(X)):
    x = X[iloop]
    x = numpy.reshape(x, (1, 1, 1))
    prediction = model.predict(x, verbose=0)
    error += abs(prediction - y[iloop])
    # print seq_in, "->", result
print('error2: %f' % error)

error = 0.0
predictions = model.predict(X, verbose=0, batch_size=len(X))
for iloop in range(len(X)):
    prediction = predictions[iloop]
    error += abs(prediction - y[iloop])
    # print seq_in, "->", result
print('error3: %f' % error)


# # demonstrate predicting random patterns
# print "Test a Random Pattern:"
# for i in range(0,20):
#     pattern_index = numpy.random.randint(len(dataX))
#     seq_in = dataX[pattern_index]
#     x = numpy.reshape(seq_in, (1, 1, 1))
#     x = x / float(256)
#     prediction = model.predict(x, verbose=0)
#     index = numpy.argmax(prediction)
#     result = prediction
#     print seq_in, "->", result
