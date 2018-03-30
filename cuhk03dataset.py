# -*- coding: utf-8 -*-
import cv2
import numpy as np

IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224

dataset_dir = "/Users/shenyi/Desktop/thesisCode/data/cuhk03_release/labeled/val/"

def get_img():
    query_img = cv2.imread("/Users/shenyi/Desktop/thesisCode/data/cuhk03_release/labeled/val/0000_00.jpg")
    query_img = cv2.resize(query_img, (IMAGE_WIDTH, IMAGE_HEIGHT))
    query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
    query_img = np.reshape(query_img, (IMAGE_HEIGHT, IMAGE_WIDTH, 3)).astype(float)
    query_imgs = []
    query_imgs.append(query_img)
    query_labels = []
    query_labels.append(0)

    test_imgs = []
    test_labels = []
    for id in range(10):
        test_img = cv2.imread(dataset_dir+"000"+str(id)+"_05.jpg")
        test_img = cv2.resize(test_img, (IMAGE_WIDTH, IMAGE_HEIGHT))
        test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
        test_img = np.reshape(test_img, (IMAGE_HEIGHT, IMAGE_WIDTH, 3)).astype(float)
        test_imgs.append(test_img)

        test_labels.append(id)
        print(id)
    return np.asarray(query_imgs, np.float32),np.asarray(test_imgs, np.float32),\
           np.asarray(query_labels, np.int32),np.asarray(test_labels, np.int32)