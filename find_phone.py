#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 19:53:38 2019

@author: jdeguzman
"""

# Runs phone detector on test images

from train_phone_finder import *
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from sys import argv

    
#def load_images_from_folder(folder):
#    images = []
#    for filename in os.listdir(folder):
#        img = cv2.imread(os.path.join(folder,filename))
#        if img is not None:
#            images.append(img)
#        return images

if __name__ == '__main__':
    testpath = argv[1]
    print(testpath)
    os.chdir('./%s' %testpath)
    #### Change this folder path to the appropriate test folder ####
    folder = './find_phone'
    
    img = cv2.imread(os.chdir('./%s' %testpath))
#    print(img)
#    for filename in os.listdir(folder):
#        if filename.endswith('.jpg'):
#            # read one test image
#            img = cv2.imread(os.path.join(folder,filename))
#            my_detector = PhoneDetector()
#            detect = my_detector.get_centroid(img)
#            print(detect[0], detect[1])
   