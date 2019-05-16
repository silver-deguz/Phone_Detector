#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 13:26:33 2019

@author: jdeguzman
"""

import logging
import numpy as np
import pickle
from matplotlib import pyplot as plt
from roipoly import RoiPoly
from PIL import Image


logger = logging.getLogger(__name__)

logging.basicConfig(format='%(levelname)s ''%(processName)-10s : %(asctime)s '
                           '%(module)s.%(funcName)s:%(lineno)s %(message)s',
                    level=logging.INFO)

# Implements RoiPoly to get color class labels for the first XX images in the
# training dataset. The color class pixel data is saved in a pickle file.

K1 = np.empty((0,3), int) # black color
K2 = np.empty((0,3), int) # light gray-whitish
K3 = np.empty((0,3), int) # dark gray-blackish


folder = '/Users/jdeguzman/Desktop/find_phone_task/find_phone'
fname = '/Users/jdeguzman/Desktop/find_phone_task/find_phone/labels.txt'

with open (fname, 'r') as myfile:
#    data = myfile.readlines()
    img = []
    labels = []
    for line in myfile:
        words = line.split()
        img.append(words[0])
        labels.append((float(words[1]), float(words[2])))

X_train, y_val = img[:100], labels[:100]

for i in range(30):
    img = X_train[i]
    I1 = np.array(Image.open('./find_phone/%s' %img), dtype='int')
    gray = I1[:,:,0]

    # Show the image
    fig = plt.figure()
    plt.imshow(I1, interpolation='nearest')
    plt.title('Image %i, draw first ROI' %i)
    plt.show(block=False)

    # Let user draw first ROI
    roi1 = RoiPoly(color='r', fig=fig)

    # Show the image
    fig = plt.figure()
    plt.imshow(I1, interpolation='nearest')
    plt.title('Image %i, draw second ROI' %i)
    plt.show(block=False)

    # Let user draw second ROI
    roi2 = RoiPoly(color='b', fig=fig)

    # Show the image
    fig = plt.figure()
    plt.imshow(I1, interpolation='nearest')
    plt.title('Image %i, draw third ROI' %i)
    plt.show(block=False)

    # Let user draw third ROI
    roi3 = RoiPoly(color='r', fig=fig)

    # Show the image
    fig = plt.figure()
    plt.imshow(I1, interpolation='nearest')
    plt.title('Image %i, draw fourth ROI' %i)
    plt.show(block=False)


    # # Show the image with both ROIs
    # plt.imshow(I1, interpolation='nearest')
    # [x.display_roi() for x in [roi1, roi2, roi3]]
    # roi1.display_roi()
    # plt.title('All ROIs')
    # plt.show()

    mask1 = roi1.get_mask(gray)
    mask2 = roi2.get_mask(gray)
    mask3 = roi3.get_mask(gray)

    # Show ROI masks
    plt.imshow(mask1 + mask2 + mask3, \
               interpolation='nearest', cmap="Greys")
    plt.title('ROI masks of the ROIs')
    plt.show()

    r1,c1 = np.where(mask1 == True)
    r2,c2 = np.where(mask2 == True)
    r3,c3 = np.where(mask3 == True)

    # 3 color classes for black, light gray-whiteish, dark gray-blackish
    K1 = np.append(K1, I1[r1,c1,:], axis=0)
    K2 = np.append(K2, I1[r2,c2,:], axis=0)
    K3 = np.append(K3, I1[r3,c3,:], axis=0)


# filename = open('phoneclasses.pkl', 'wb')
# pickle.dump(K1, filename)
# pickle.dump(K2, filename)
# pickle.dump(K3, filename)
# filename.close()
