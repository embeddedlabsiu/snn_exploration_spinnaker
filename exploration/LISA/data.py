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

def data():
    """Data preprocessing

    It checks the GTSRB folder and pre-processes the data.
    In order to reduce time, we use pickle to create ready files.

    Returns:
        [X_train] -- [X data used for training]
        [Y_train] -- [Y data used for training]
        [X_test] -- [X data used for testing]
        [Y_test] -- [Y data used for testing]
    """

    try:
        X_train = pickle.load(open("X_train.p", "rb"))
        Y_train = pickle.load(open("Y_train.p", "rb"))
    except (OSError, IOError):
        X_train, Y_train = create_training_set()
        X_train_pickle = open("X_train.p", "wb")
        pickle.dump(X_train, X_train_pickle)
        Y_train_pickle = open("Y_train.p", "wb")
        pickle.dump(Y_train, Y_train_pickle)
    try:
        X_test = pickle.load(open("X_test.p", "rb"))
        Y_test = pickle.load(open("Y_test.p", "rb"))
    except (OSError, IOError):
        X_test, Y_test = create_test_set()
        X_test_pickle = open("X_test.p", "wb")
        pickle.dump(X_test, X_test_pickle)
        Y_test_pickle = open("Y_test.p", "wb")
        pickle.dump(Y_test, Y_test_pickle)

    # convert class vectors to binary class matrices
    # X_train = X_train.astype('float32')
    # X_test = X_test.astype('float32')
    # X_train /= 255
    # X_test /= 255

    # # this will do preprocessing and realtime data augmentation
    # datagen = ImageDataGenerator(
    #     featurewise_center=False,  # set input mean to 0 over the dataset
    #     samplewise_center=False,  # set each sample mean to 0
    #     featurewise_std_normalization=False,  # divide inputs by std of the dataset
    #     samplewise_std_normalization=False,  # divide each input by its std
    #     zca_whitening=False,  # apply ZCA whitening
    #     rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
    #     width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    #     height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    #     horizontal_flip=True,  # randomly flip images
    #     vertical_flip=False)  # randomly flip images

    # # compute quantities required for featurewise normalization
    # # (std, mean, and principal components if ZCA whitening is applied)
    # datagen.fit(X_train)
    return X_train, Y_train, X_test, Y_test


def GTSRB_data():
    """Data preprocessing

    It checks the GTSRB folder and pre-processes the data.
    In order to reduce time, we use pickle to create ready files.

    Returns:
        [X_train] -- [X data used for training]
        [Y_train] -- [Y data used for training]
        [X_test] -- [X data used for testing]
        [Y_test] -- [Y data used for testing]
    """
    print("Performing analysis on GTSRB dataset")
    NUM_CLASSES = 43
    IMG_SIZE = 28   #default = 48

    X_train, Y_train = create_training_set(IMG_SIZE)
    # X_test, Y_test = create_test_set(IMG_SIZE)
    X_test, Y_test = create_test_set_phone(IMG_SIZE)

    
    if K.image_dim_ordering() == 'th':
        X_train = np.array([cv2.resize(img.transpose(1,2,0), (IMG_SIZE,IMG_SIZE)).transpose(2,0,1) for img in X_train])
        X_test = np.array([cv2.resize(img.transpose(1,2,0), (IMG_SIZE,IMG_SIZE)).transpose(2,0,1) for img in X_test])
    else:
        X_train = np.array([cv2.resize(img, (IMG_SIZE, IMG_SIZE)) for img in X_train])
        X_test = np.array([cv2.resize(img, (IMG_SIZE, IMG_SIZE)) for img in X_test])

    np.savez('./dataset/x_norm.npz', X_train)
    np.savez('./dataset/x_test.npz', X_test)
    np.savez('./dataset/y_test.npz', Y_test)

    input_shape = (3, IMG_SIZE, IMG_SIZE)
    return X_train, Y_train, X_test, Y_test, NUM_CLASSES, IMG_SIZE, input_shape

def GTSRB_grayscale_data(img_size=48):
    """Data preprocessing

    It checks the GTSRB folder and pre-processes the data.
    In order to reduce time, we use pickle to create ready files.

    Returns:
        [X_train] -- [X data used for training]
        [Y_train] -- [Y data used for training]
        [X_test] -- [X data used for testing]
        [Y_test] -- [Y data used for testing]
    """
    print("Performing analysis on GTSRB dataset")
    NUM_CLASSES = 43

    X_train, Y_train = create_training_set(img_size)
    X_test, Y_test = create_test_set(img_size)
    # X_test, Y_test = create_test_set_phone(img_size)

    if K.image_dim_ordering() == 'th':
        X_train = X_train.reshape(X_train.shape[0], 3, img_size, img_size)
        X_test = X_test.reshape(X_test.shape[0], 3, img_size, img_size)

    # if K.image_dim_ordering() == 'th':
    #     X_train = X_train.reshape(X_train.shape[0], 1, img_size, img_size)
    #     X_test = X_test.reshape(X_test.shape[0], 1, img_size, img_size)

    np.savez('x_norm.npz', X_train)
    np.savez('x_test.npz', X_test)
    np.savez('y_test.npz', Y_test)

    # exit()

    # input_shape = (1, img_size, img_size)
    input_shape = (3, img_size, img_size)
    return X_train, Y_train, X_test, Y_test, NUM_CLASSES, img_size, input_shape

def cifar10_data(IMG_SIZE=32):

    print("Performing analysis on CIFAR 10 dataset")
    NUM_CLASSES = 10
    # IMG_SIZE = 32  #default
    # IMG_SIZE = 64
    subtract_pixel_mean = True

    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()


    if K.image_dim_ordering() == 'th':
        X_train = np.array([cv2.resize(img.transpose(1,2,0), (IMG_SIZE,IMG_SIZE)).transpose(2,0,1) for img in X_train])
        X_test = np.array([cv2.resize(img.transpose(1,2,0), (IMG_SIZE,IMG_SIZE)).transpose(2,0,1) for img in X_test])
    else:
        X_train = np.array([cv2.resize(img, (IMG_SIZE, IMG_SIZE)) for img in X_train])
        X_test = np.array([cv2.resize(img, (IMG_SIZE, IMG_SIZE)) for img in X_test])

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    if subtract_pixel_mean:
        x_train_mean = np.mean(X_train, axis=0)
        X_train -= x_train_mean
        X_test -= x_train_mean

    Y_train = keras.utils.to_categorical(Y_train, NUM_CLASSES)
    Y_test = keras.utils.to_categorical(Y_test, NUM_CLASSES)

    np.savez('x_norm.npz', X_train)
    np.savez('x_test.npz', X_test)
    np.savez('y_test.npz', Y_test)

    input_shape = (3, IMG_SIZE, IMG_SIZE)
    return X_train, Y_train, X_test, Y_test, NUM_CLASSES, IMG_SIZE, input_shape


def cifar100_data():
    print("Performing analysis on CIFAR 100 dataset")
    NUM_CLASSES = 100
    IMG_SIZE = 32
    subtract_pixel_mean = True

    (X_train, Y_train), (X_test, Y_test) = cifar100.load_data()

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    if subtract_pixel_mean:
        x_train_mean = np.mean(X_train, axis=0)
        X_train -= x_train_mean
        X_test -= x_train_mean

    Y_train = keras.utils.to_categorical(Y_train, NUM_CLASSES)
    Y_test = keras.utils.to_categorical(Y_test, NUM_CLASSES)



    input_shape = (3, IMG_SIZE, IMG_SIZE)

    return X_train, Y_train, X_test, Y_test, NUM_CLASSES, IMG_SIZE, input_shape


def mnist_data():
    print("Performing analysis on MNIST dataset")
    NUM_CLASSES = 10
    IMG_SIZE = 28
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()

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
    print(X_train.shape)
    print(X_test.shape)

    return X_train, Y_train, X_test, Y_test, NUM_CLASSES, IMG_SIZE, input_shape

def mnist_data_image_size(IMG_SIZE = 28):

    print("Performing analysis on MNIST dataset")
    NUM_CLASSES = 10
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()

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

def mnist_data_segments(IMG_SIZE = 28):

    print("Breaks MNIST dataset into segments of 10 images each")
    NUM_CLASSES = 10
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()

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
    segments = 1000
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
        save_dir='./MNIST_segments/'+str(IMG_SIZE)+'/'
        np.savez(save_dir+'x_test_'+str(segment_cnt)+'.npz', X_test_seg)
        np.savez(save_dir+'y_test_'+str(segment_cnt)+'.npz', y_test_seg)

        X_test_seg = []
        y_test_seg = []


    np.savez(save_dir+'x_norm.npz', X_train)
    

    input_shape = (1, IMG_SIZE, IMG_SIZE)
    return X_train, Y_train, X_test, Y_test, NUM_CLASSES, IMG_SIZE, input_shape


def LISA_data(img_size=48):
    """Data preprocessing

    It checks the GTSRB folder and pre-processes the data.
    In order to reduce time, we use pickle to create ready files.

    Returns:
        [X_train] -- [X data used for training]
        [Y_train] -- [Y data used for training]
        [X_test] -- [X data used for testing]
        [Y_test] -- [Y data used for testing]
    """
    print("Performing analysis on LISA dataset")
    NUM_CLASSES = 28

    X_train, Y_train = LISA_create_training_set(img_size)
    X_test, Y_test = LISA_create_test_set(img_size)
    # X_test, Y_test = LISA_create_test_set_phone(img_size)

    if K.image_dim_ordering() == 'th':
        X_train = X_train.reshape(X_train.shape[0], 3, img_size, img_size)
        X_test = X_test.reshape(X_test.shape[0], 3, img_size, img_size)

    # print 'X_train shape:', X_train.shape
    # print 'X_test shape:', X_test.shape

    np.savez('x_norm.npz', X_train)
    np.savez('x_test.npz', X_test)
    np.savez('y_test.npz', Y_test)

    # input_shape = (1, img_size, img_size)
    input_shape = (3, img_size, img_size)
    return X_train, Y_train, X_test, Y_test, NUM_CLASSES, img_size, input_shape

def LISA_data_segment(img_size=48):
    """

    Creates segments of LISA test set to accelerate experiments on SpiNNaker
    Total number of images: 1374
    Each segment contains 10 images

    """
    print("Performing analysis on LISA dataset")
    NUM_CLASSES = 28

    X_train, Y_train = LISA_create_training_set(img_size)
    X_test, Y_test = LISA_create_test_set_segment(img_size)
    

    if K.image_dim_ordering() == 'th':
        X_train = X_train.reshape(X_train.shape[0], 3, img_size, img_size)

    np.savez('x_norm.npz', X_train)
    input_shape = (3, img_size, img_size)
    return X_train, Y_train, X_test, Y_test, NUM_CLASSES, img_size, input_shape

def GTSRB_data_segment(img_size=48):
    """

    Creates segments of GTSRB test set to accelerate experiments on SpiNNaker
    Total number of images: 1370
    Each segment contains 10 images

    """
    
    print("Performing analysis on GTSRB dataset")
    NUM_CLASSES = 43

    X_train, Y_train = create_training_set(img_size, NUM_CLASSES)
    X_test, Y_test = GTSRB_create_test_set_segment(img_size,NUM_CLASSES)


    if K.image_dim_ordering() == 'th':
        X_train = np.array([cv2.resize(img.transpose(1,2,0), (img_size,img_size)).transpose(2,0,1) for img in X_train])
        # X_test = np.array([cv2.resize(img.transpose(1,2,0), (img_size,img_size)).transpose(2,0,1) for img in X_test])
    else:
        X_train = np.array([cv2.resize(img, (img_size, img_size)) for img in X_train])
        # X_test = np.array([cv2.resize(img, (img_size, img_size)) for img in X_test])


    np.savez('x_norm.npz', X_train)
    input_shape = (3, img_size, img_size)
    return X_train, Y_train, X_test, Y_test, NUM_CLASSES, img_size, input_shape

def cifar10_data_segment(IMG_SIZE=32):
    """

    Creates segments of CIFAR10 test set to accelerate experiments on SpiNNaker
    Total number of images: 1300 (total CIFAR10 test set: 10,000 images)
    Each segment contains 10 images

    """
    print("Performing analysis on CIFAR 10 dataset")
    NUM_CLASSES = 10
    subtract_pixel_mean = True

    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()


    if K.image_dim_ordering() == 'th':
        X_train = np.array([cv2.resize(img.transpose(1,2,0), (IMG_SIZE,IMG_SIZE)).transpose(2,0,1) for img in X_train])
        X_test = np.array([cv2.resize(img.transpose(1,2,0), (IMG_SIZE,IMG_SIZE)).transpose(2,0,1) for img in X_test])
    else:
        X_train = np.array([cv2.resize(img, (IMG_SIZE, IMG_SIZE)) for img in X_train])
        X_test = np.array([cv2.resize(img, (IMG_SIZE, IMG_SIZE)) for img in X_test])

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    if subtract_pixel_mean:
        x_train_mean = np.mean(X_train, axis=0)
        X_train -= x_train_mean
        X_test -= x_train_mean

    Y_train = keras.utils.to_categorical(Y_train, NUM_CLASSES)
    Y_test = keras.utils.to_categorical(Y_test, NUM_CLASSES)

    # create segments for x_test and y_test
    segments = 130
    print("Current directory test ",os.getcwd())

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
        save_dir='./CIFAR10_segments/'+str(IMG_SIZE)+'/'
        np.savez(save_dir+'x_test_'+str(segment_cnt)+'.npz', X_test_seg)
        np.savez(save_dir+'y_test_'+str(segment_cnt)+'.npz', y_test_seg)

        X_test_seg = []
        y_test_seg = []


    np.savez(save_dir+'x_norm.npz', X_train)
    
    # np.savez('x_test.npz', X_test)
    # np.savez('y_test.npz', Y_test)

    input_shape = (3, IMG_SIZE, IMG_SIZE)
    return X_train, Y_train, X_test, Y_test, NUM_CLASSES, IMG_SIZE, input_shape

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

