import tensorflow as tf
mnist=tf.keras.datasets.mnist
import numpy as np

(x_train,y_train), (x_test,y_test)=mnist.load_data()
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
images = np.zeros((x_train.shape[0] + x_test.shape[0], x_train.shape[1], x_train.shape[2]))
images[:x_train.shape[0]] = x_train[:]
images[x_train.shape[0]:] = x_test[:]
print(images.shape)
np.save('./images.npy', images)

labels = np.zeros((y_train.shape[0] + y_test.shape[0], ))
labels[:y_train.shape[0]] = y_train[:]
labels[y_train.shape[0]:] = y_test[:]
print(labels.shape)
np.save('./labels.npy', labels)
exit()

x_train,x_test=x_train/255.0, x_test/255.0
model=tf.keras.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)