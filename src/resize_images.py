import os
import sys
# from PIL import Image
from typing import Union

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from skimage import io
from skimage.transform import resize
import numpy as np
from multiprocessing import pool
from multiprocessing.dummy import Pool as ThreadPool


def create_directory(directory):
    '''
    Creates a new folder in the specified directory if the folder doesn't exist.

    INPUT
        directory: Folder to be created, called as "folder/".

    OUTPUT
        New folder in the current directory.
    '''
    if not os.path.exists(directory):
        os.makedirs(directory)


def crop_and_resize_images(path, new_path, cropx, cropy, img_size=256):
    '''
    Crops, resizes, and stores all images from a directory in a new directory.

    INPUT
        path: Path where the current, unscaled images are contained.
        new_path: Path to save the resized images.
        img_size: New size for the rescaled images.

    OUTPUT
        All images cropped, resized, and saved from the old folder to the new folder.
    '''
    create_directory(new_path)
    dirs = [l for l in os.listdir(path) if l != '.DS_Store']
    total = 0
    images = []

    for item in dirs:
        images.append(item)

    def resize_save_image(item):
        img = io.imread(path + item)
        y, x, channel = img.shape
        startx = x // 2 - (cropx // 2)
        starty = y // 2 - (cropy // 2)
        img = img[starty:starty + cropy, startx:startx + cropx]
        img = resize(img, (256,256))
        io.imsave(str(new_path + item), img)
        print("Saving: ", item)

    pool = ThreadPool()
    pool.map(resize_save_image, images)

    ## Single threaded implementation
    # for item in dirs:
    #     img = io.imread(path+item)
    #     y,x,channel = img.shape
    #     startx = x//2-(cropx//2)
    #     starty = y//2-(cropy//2)
    #     img = img[starty:starty+cropy,startx:startx+cropx]
    #     img = resize(img, (256,256))
    #     io.imsave(str(new_path + item), img)
    #     total += 1
    #     print("Saving: ", item, total)


if __name__ == '__main__':
    crop_and_resize_images(path='D:/temp/data/train/', new_path='../data/train-resized-256/', cropx=1800, cropy=1800, img_size=256)
    crop_and_resize_images(path='D:/temp/data/test/', new_path='../data/test-resized-256/', cropx=1800, cropy=1800, img_size=256)
