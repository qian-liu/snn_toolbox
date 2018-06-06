# coding=utf-8

"""LeNet for MNIST"""
import numpy as np
import os
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, AveragePooling2D, MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, TensorBoard

from keras.constraints import min_max_norm

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session



config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

np.random.seed(0)

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32') / 255.
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32') / 255.
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

nonlinearity = 'relu'

model = Sequential()

model.add(Conv2D(16, (5, 5), input_shape=(1, 28, 28), activation=nonlinearity, use_bias=False)) #bias_constraint=min_max_norm(min_value=0.0, max_value=0.0)))
model.add(AveragePooling2D())

model.add(Conv2D(64, (5, 5), activation=nonlinearity, use_bias=False)) #bias_constraint=min_max_norm(min_value=0.0, max_value=0.0)))
model.add(AveragePooling2D())
model.add(Dropout(0.5))

# model.add(Conv2D(120, (5, 5), padding='same', activation=nonlinearity))

model.add(Flatten())
model.add(Dense(10, activation=nonlinearity, use_bias=False)) #bias_constraint=min_max_norm(min_value=0.0, max_value=0.0)))
# model.add(Dense(10, activation='softmax'))

model.compile('adam', 'mean_squared_error', metrics=['accuracy'])

path = '.'

if not os.path.exists(os.path.join(path, 'weights')):
    os.makedirs(os.path.join(path, 'weights'))
checkpoint = ModelCheckpoint('weights/weights.{epoch:02d}-{val_acc:.2f}.h5', 'val_acc')
gradients = TensorBoard(os.path.join(path, 'logs'), 2, write_grads=True)

# verbose: Integer. 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
model.fit(X_train, Y_train, batch_size=256, epochs=20, verbose=2, validation_data=(X_test, Y_test),
          callbacks=[checkpoint, gradients])

score = model.evaluate(X_test, Y_test, verbose=2)
print('Test score:', score[0])
print('Test accuracy:', score[1])

model.save(os.path.join(path, '{:2.2f}.h5'.format(score[1]*100)))
# tensorboard --logdir=logs
