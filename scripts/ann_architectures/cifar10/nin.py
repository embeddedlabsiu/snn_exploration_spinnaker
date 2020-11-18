# -*- coding: utf-8 -*-
"""

see http://arxiv.org/pdf/1312.4400v3.pdf

Should get to 8.81% error when using data-augmentation (10.41% without).

Apply ZCA whitening and GCN.

Created on Fri Aug 19 09:15:25 2016

@author: rbodo
"""


from __future__ import absolute_import
from __future__ import print_function

from keras.callbacks import ModelCheckpoint
from keras.datasets import cifar10
from keras.layers.convolutional import Conv2D, AveragePooling2D
from keras.layers.core import Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras.utils import np_utils

from snntoolbox.simulation.plotting import plot_history

batch_size = 128
nb_classes = 10
nb_epoch = 200

# Input image dimensions
img_rows, img_cols = 32, 32
img_channels = 3

# Data set
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

init = 'he_uniform'
reg = l2(0.0001)
b_reg = None

model = Sequential()

model.add(Conv2D(192, 5, 5, padding='same', init=init,
                        input_shape=(img_channels, img_rows, img_cols)))
model.add(BatchNormalization(axis=1))
model.add(Activation('relu'))
model.add(Conv2D(160, 1, 1, W_regularizer=reg, b_regularizer=b_reg,
                        init=init))
model.add(BatchNormalization(axis=1))
model.add(Activation('relu'))
model.add(Conv2D(96, 1, 1, W_regularizer=reg, b_regularizer=b_reg,
                        init=init))
model.add(BatchNormalization(axis=1))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2),
                           padding='same'))
model.add(Dropout(0.2))

model.add(Conv2D(192, 5, 5, padding='same', init=init,
                        W_regularizer=reg, b_regularizer=b_reg))
model.add(BatchNormalization(axis=1))
model.add(Activation('relu'))
model.add(Conv2D(192, 1, 1, W_regularizer=reg, b_regularizer=b_reg,
                        init=init))
model.add(BatchNormalization(axis=1))
model.add(Activation('relu'))
model.add(Conv2D(192, 1, 1, W_regularizer=reg, b_regularizer=b_reg,
                        init=init))
model.add(BatchNormalization(axis=1))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2),
                           padding='same'))
model.add(Dropout(0.2))

model.add(Conv2D(192, 3, 3, padding='same', init=init,
                        W_regularizer=reg, b_regularizer=b_reg))
model.add(BatchNormalization(axis=1))
model.add(Activation('relu'))
model.add(Conv2D(192, 1, 1, W_regularizer=reg, b_regularizer=b_reg,
                        init=init))
model.add(BatchNormalization(axis=1))
model.add(Activation('relu'))
model.add(Conv2D(10, 1, 1, W_regularizer=reg, b_regularizer=b_reg,
                        init=init))
model.add(BatchNormalization(axis=1))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(8, 8), strides=(1, 1)))
model.add(Flatten())
model.add(Activation('softmax'))

model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])

# Whether to apply global contrast normalization and ZCA whitening
gcn = True
zca = True

traingen = ImageDataGenerator(rescale=1./255, featurewise_center=gcn,
                              featurewise_std_normalization=gcn,
                              zca_whitening=zca, horizontal_flip=True,
                              rotation_range=10, width_shift_range=0.1,
                              height_shift_range=0.1)

# Compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
traingen.fit(X_train/255.)

trainflow = traingen.flow(X_train, Y_train, batch_size=batch_size)

testgen = ImageDataGenerator(rescale=1./255, featurewise_center=gcn,
                             featurewise_std_normalization=gcn,
                             zca_whitening=zca)

testgen.fit(X_test/255.)

testflow = testgen.flow(X_test, Y_test, batch_size=batch_size)

checkpointer = ModelCheckpoint(filepath='nin.{epoch:02d}-{val_acc:.2f}.h5',
                               verbose=1, save_best_only=True)

# Fit the model on the batches generated by datagen.flow()
history = model.fit_generator(trainflow, nb_epoch=nb_epoch,
                              samples_per_epoch=len(X_train),
                              validation_data=testflow,
                              nb_val_samples=len(X_test),
                              callbacks=[checkpointer])
plot_history(history)

score = model.evaluate_generator(testflow, val_samples=len(X_test))
print('Test score:', score[0])
print('Test accuracy:', score[1])

model.save('{:2.2f}.h5'.format(score[1]*100))
