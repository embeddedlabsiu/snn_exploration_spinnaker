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

def LISA_create_training_set(img_size=48, num_classes=28):
    root_dir = 'LISA_dataset/Final_Training/Images/'
    imgs = []
    labels = []

    all_img_paths = glob.glob(os.path.join(root_dir, '*/*.png'))
    print("Current directory train ",os.getcwd())
    np.random.shuffle(all_img_paths)

    i = 0
    limit = 10000
    for img_path in all_img_paths:
        original = io.imread(img_path)
        img = preprocess_img(io.imread(img_path), img_size)
        label = get_class(img_path)
        imgs.append(img)
        labels.append(label)
        i = i+1
        if (i == limit):
            break

    X = np.array(imgs)
    
    # Make one hot targets
    Y = np.eye(num_classes, dtype='uint8')[labels]

    return X, Y
