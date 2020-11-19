import numpy as np
from skimage import color, exposure, transform
from skimage import io
import os, time, sys
import pandas as pd
from pre_process_img import *
import keras
from skimage.color import rgb2gray
from keras import backend as K

def LISA_create_test_set(img_size=48, num_classes=28):
    print("Current directory test ",os.getcwd())

    # Load test dataset
    test = pd.read_csv('LISA_dataset/LISA-final_test.csv', sep=';')
    X_test = []
    y_test = []
    i = 0
    limit = 1374
    for file_name, class_id in zip(list(test['Filename']), list(test['ClassId'])):
        img_path = os.path.join('LISA_dataset/Final_Test/Images/', file_name)
        img = preprocess_img(io.imread(img_path), img_size)
        X_test.append(img)
        y_test.append(class_id)
        i = i+1
        if (i == limit):
            break

    X_test = np.array(X_test)

    y_test = keras.utils.to_categorical(y_test, num_classes)

    return X_test, y_test

def LISA_create_test_set_segment(img_size=48, num_classes=28):
    print("Current directory test ",os.getcwd())

    # Load test dataset
    test = pd.read_csv('LISA_dataset/LISA-final_test.csv', sep=';')
    X_test = []
    y_test = []
    i = 0
    #limit = 1000
    limit = 1374
    cnt = 0
    segment_cnt = 0
    for file_name, class_id in zip(list(test['Filename']), list(test['ClassId'])):
        img_path = os.path.join('LISA_dataset/Final_Test/Images/', file_name)
        img = preprocess_img(io.imread(img_path), img_size)
        X_test.append(img)
        y_test.append(class_id)
        i = i+1
        cnt += 1
        if (cnt == 10 or i == limit):
            X_test = np.array(X_test)
            y_test = keras.utils.to_categorical(y_test, num_classes)
            if K.image_dim_ordering() == 'th':
                X_test = X_test.reshape(X_test.shape[0], 3, img_size, img_size)
            segment_cnt += 1
            np.savez('x_test_'+str(segment_cnt)+'.npz', X_test)
            np.savez('y_test_'+str(segment_cnt)+'.npz', y_test)
            cnt = 0
            X_test = []
            y_test = []

        if (i == limit):
            break

    return X_test, y_test


