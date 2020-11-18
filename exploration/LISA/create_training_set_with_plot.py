import os
from skimage import io
import glob
import numpy as np
from pre_process_img import *
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.color import rgb2gray
from skimage import color, exposure, transform

def get_class(img_path):
    return int(img_path.split('/')[-2])

def create_training_set(img_size=48, num_classes=43):
    root_dir = 'GTSRB/Final_Training/Images/'
    imgs = []
    labels = []

    all_img_paths = glob.glob(os.path.join(root_dir, '*/*.ppm'))
    print("Current directory train ",os.getcwd())
    np.random.shuffle(all_img_paths)

    i = 0
    limit = 3
    for img_path in all_img_paths:
        original = io.imread(img_path)
        img = grayscale_img(io.imread(img_path), img_size)

        gray = []
        f = plt.figure()
        f.add_subplot(1, 2, 1)
        plt.imshow(original)
        f.add_subplot(1, 2, 2)
        plt.imshow(img, cmap='gray')
        plt.show(block=True)

        ORIGINAL_IMG = np.array(original)
        gray.append(img)
        GRAY_IMG = np.array(gray)
        print (ORIGINAL_IMG)
        print (GRAY_IMG)




        # plt.figure()
        # plt.imshow(original)
        # plt.imshow(img, cmap='gray')
        # plt.show()

        label = get_class(img_path)
        imgs.append(img)
        labels.append(label)
        i = i+1
        if (i == limit):
            break

    X = np.array(imgs)
    # print X.shape
    
    # Make one hot targets
    Y = np.eye(num_classes, dtype='uint8')[labels]
    exit()
    return X, Y
