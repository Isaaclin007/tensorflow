# -*- coding:UTF-8 -*-

from __future__ import print_function
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras.models import Sequential
# from tensorflow.keras.utils.vis_utils import plot_model
import numpy as np


def process_txt(open_path):
    with open(open_path, 'rb') as f:
        lines = []
        for line in f:
            line = line.strip().lower()
            line = line.decode("ascii", "ignore")
            if 0 == len(line):
                continue
            lines.append(line)
    text = ' '.join(lines)
    return text


text = process_txt('./data/test.txt')

print("len(text): %u" % len(text))
# print(text)

chars = set([c for c in text])
chars_count = len(chars)
char2index = dict((c, i) for i, c in enumerate(chars))
index2char = dict((i, c) for i, c in enumerate(chars))

# print("char2index:")
# print(char2index)

SEQLEN = 10
STEP = 1
input_chars = []
label_chars = []
for  i in range(0, len(text) - SEQLEN, STEP):
    input_chars.append(text[i:i+SEQLEN])
    label_chars.append(text[i+SEQLEN])

X = np.zeros((len(input_chars), SEQLEN, chars_count), dtype=np.bool)
Y = np.zeros((len(input_chars), chars_count), dtype=np.bool)
for i,input_char in enumerate(input_chars):
    for j,c in enumerate(input_char):
        X[i, j, char2index[c]] = 1
    Y[i, char2index[label_chars[i]]] = 1

print("X.shape:")
print(X.shape)

print("Y.shape:")
print(Y.shape)

HIDDEN_SIZE = 128
BATCH_SIZE = 128
NUM_ITERATIONS = 1
NUM_EPOCHS_PER_ITERATION = 1
NUM_PREDS_PER_EPOCH = 100

model = Sequential()
model.add(SimpleRNN(HIDDEN_SIZE, return_sequences=False,
                    input_shape=(SEQLEN, chars_count),unroll=True))
model.add(Dense(chars_count))
model.add(Activation("softmax"))
model.compile(loss="categorical_crossentropy", optimizer="rmsprop")
model.summary()

for iteration in range(NUM_ITERATIONS):
    print('Iteration : %d'%iteration)
    model.fit(X, Y, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS_PER_ITERATION)
    # 训练1epoch,测试一次
    test_idx = np.random.randint(len(input_chars))
    test_chars = input_chars[test_idx]
    print('test seed is : %s'%test_chars)
    print(test_chars,end='')
    for i in range(NUM_PREDS_PER_EPOCH):
        # 测试序列向量化
        vec_test = np.zeros((1, SEQLEN, chars_count))
        for i, ch in enumerate(test_chars):
            vec_test[0, i, char2index[ch]] = 1
        pred = model.predict(vec_test, verbose=0)[0]
        pred_char = index2char[np.argmax(pred)]
        print(pred_char,end='')
        # 不断的加入新生成字符组成新的序列
        test_chars = test_chars[1:] + pred_char
    print('\n')
