import pickle
import cv2
import os
from keras import backend as K
import keras
from create_test_set import create_test_set, create_test_set_phone, LISA_create_test_set, LISA_create_test_set_phone, LISA_create_test_set_segment, GTSRB_create_test_set_segment
from create_training_set import create_training_set, LISA_create_training_set
from keras.datasets import cifar100, cifar10, mnist, fashion_mnist
import numpy as np
from skimage.transform import resize
def fashion_mnist_data_image_size(IMG_SIZE = 28):

    print("Performing analysis on Fasion MNIST dataset")
    NUM_CLASSES = 10
    (X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()

    if IMG_SIZE!=28:
        X_train = np.array([cv2.resize(img, (IMG_SIZE,IMG_SIZE)) for img in X_train])
        X_test = np.array([cv2.resize(img, (IMG_SIZE,IMG_SIZE)) for img in X_test])
    
    X_train = X_train.reshape(X_train.shape[0], 1, IMG_SIZE, IMG_SIZE)
    X_test = X_test.reshape(X_test.shape[0], 1, IMG_SIZE, IMG_SIZE)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    Y_train = keras.utils.to_categorical(Y_train, NUM_CLASSES)
    Y_test = keras.utils.to_categorical(Y_test, NUM_CLASSES)

    np.savez('x_norm.npz', X_train)
    np.savez('x_test.npz', X_test)
    np.savez('y_test.npz', Y_test)

    input_shape = (1, IMG_SIZE, IMG_SIZE)

    return X_train, Y_train, X_test, Y_test, NUM_CLASSES, IMG_SIZE, input_shape
def fashion_mnist_data_segments(IMG_SIZE = 28):

    print("Breaks Fasion MNIST dataset into segments of 10 images each")
    NUM_CLASSES = 10
    (X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()

    X_train = np.array([cv2.resize(img, (IMG_SIZE,IMG_SIZE)) for img in X_train])
    X_test = np.array([cv2.resize(img, (IMG_SIZE,IMG_SIZE)) for img in X_test])

    X_train = X_train.reshape(X_train.shape[0], 1, IMG_SIZE, IMG_SIZE)
    X_test = X_test.reshape(X_test.shape[0], 1, IMG_SIZE, IMG_SIZE)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    Y_train = keras.utils.to_categorical(Y_train, NUM_CLASSES)
    Y_test = keras.utils.to_categorical(Y_test, NUM_CLASSES)

    # segments
    segments = 130
    X_test_seg = []
    y_test_seg = []
    i = 0
    segment_cnt = 0

    for i in range(0,segments):

        cnt = 0
        while (cnt<10):
            X_test_seg.append(X_test[i])
            y_test_seg.append(Y_test[i])
            cnt  = cnt + 1

        segment_cnt  = segment_cnt + 1
        save_dir='./FMNIST_segments/'+str(IMG_SIZE)+'/'
        np.savez(save_dir+'x_test_'+str(segment_cnt)+'.npz', X_test_seg)
        np.savez(save_dir+'y_test_'+str(segment_cnt)+'.npz', y_test_seg)

        X_test_seg = []
        y_test_seg = []


    np.savez(save_dir+'x_norm.npz', X_train)
    

    input_shape = (1, IMG_SIZE, IMG_SIZE)
    return X_train, Y_train, X_test, Y_test, NUM_CLASSES, IMG_SIZE, input_shape

