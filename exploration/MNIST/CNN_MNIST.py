# make dynamic memory allocation
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config = config)

from data import *
from keras import backend as K
import numpy as np
import sys
import keras
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.optimizers import SGD, Adadelta
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
import gc
import argparse

# Dataset selection
K.set_image_data_format('channels_first')

parser = argparse.ArgumentParser()
parser.add_argument("image_size", help="image size")
parser.add_argument("stages", help="stages")
parser.add_argument("depth", help="number of building blocks per stage")
parser.add_argument("width", help="kernels used")
args = parser.parse_args()

X_train, Y_train, X_test, Y_test, NUM_CLASSES, IMG_SIZE, INPUT_SIZE = mnist_data_image_size(int(args.image_size))


batch_size = 32
epochs = 100

model = Sequential()

for s in range(1, (int(args.stages)+1)):
    for d in range(1, (int(args.depth)+1)):
        model.add(Conv2D(int(args.width)*pow(2, s-1), use_bias=False, kernel_size=(5, 5), activation='relu', input_shape=INPUT_SIZE))
    model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dropout(0.3))
model.add(Dense(256, use_bias=False, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(NUM_CLASSES, use_bias=False, activation='softmax'))

model.summary()

lr = 0.01
sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              # optimizer=SGD(),
              optimizer=Adadelta(),
              metrics=['accuracy'])


def lr_schedule(epoch):
    return lr * (0.1 ** int(epoch / 10))


model.fit(X_train, Y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=2,
          validation_data=(X_test, Y_test),
          callbacks=[ModelCheckpoint('MNIST_cnn.h5', save_best_only=True)])

score = model.evaluate(X_test, Y_test, verbose=0)

# model.save('LISA_cnn.h5')

print('Test loss:', score[0])
print('Test accuracy:', score[1])

