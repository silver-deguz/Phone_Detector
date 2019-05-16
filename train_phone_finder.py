#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 20:07:26 2019

@author: jdeguzman
"""

import numpy as np
import cv2
import os
import pickle
import time
import matplotlib.pyplot as plt

#%%    
def compute_gaussian_params(X, diagonal=False):
    mu = np.sum(X, axis=0) / len(X)
    diff = X - mu
    if diagonal:
        covar = np.zeros((3,3))
        for i in range(len(diff)):
            covar += np.diag(diff[i])**2
        covar /= len(X)
    else:
        covar = (np.transpose(diff) @ diff) / len(X)
    return mu, covar

def train(K1, K2, K3):
    mu_black, covar_black = compute_gaussian_params(K1, False)
    mu_lgray, covar_lgray = compute_gaussian_params(K2, False)
    mu_dgray, covar_dgray = compute_gaussian_params(K3, False)

    mu = [mu_black, mu_lgray, mu_dgray]
    covar = [covar_black, covar_lgray, covar_dgray]

    a_black = np.log( 1 / np.sqrt( ((2*np.pi)**3) * np.linalg.det(covar[0]) ) )
    a_lgray = np.log( 1 / np.sqrt( ((2*np.pi)**3) * np.linalg.det(covar[1]) ) )
    a_dgray = np.log( 1 / np.sqrt( ((2*np.pi)**3) * np.linalg.det(covar[2]) ) )
    a = [a_black, a_lgray, a_dgray]
    return mu, covar, a

#%%
class PhoneDetector():
    def __init__(self):
        '''
            Initializes phone detector with gaussian parameters of 
            pretrained color classfier models
        '''
        # Load color classes for detection task
        colorclasses = open('./phoneclasses.pkl', 'rb')
        K1 = pickle.load(colorclasses)
        K2 = pickle.load(colorclasses)
        K3 = pickle.load(colorclasses)
        colorclasses.close()
        
        # Compute mean, covariance, and a
        mu, covar, a = train(K1, K2, K3)
        self.mu = mu
        self.covar = covar
        self.a = a
    
    def segment_image(self, img):
        '''
            Calculate the segmented image using a Single Gaussian classifier
            
            Inputs:
				img - original image
            Outputs:
				mask_img - a binary image with 1 if the pixel in the original 
                image is black and 0 otherwise
        '''
        
        test_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        X = test_img.reshape((img.shape[0]*img.shape[1]), img.shape[2])
        
        # Color classification
        probs = np.zeros((len(X),3)) # using 3 color classes
        diff0 = X - self.mu[0]
        diff1 = X - self.mu[1]
        diff2 = X - self.mu[2]
        
        diff0T = np.transpose(diff0)
        diff1T = np.transpose(diff1)
        diff2T = np.transpose(diff2)
        
        inv_cov0 = np.linalg.inv(self.covar[0])
        inv_cov1 = np.linalg.inv(self.covar[1])
        inv_cov2 = np.linalg.inv(self.covar[2])
        
        for i in range(len(X)):
            probs[i,0] = (-0.5 * (diff0[i,:] @ inv_cov0 @ diff0T[:,i])) + self.a[0]
            probs[i,1] = (-0.5 * (diff1[i,:] @ inv_cov1 @ diff1T[:,i])) + self.a[1]
            probs[i,2] = (-0.5 * (diff2[i,:] @ inv_cov2 @ diff2T[:,i])) + self.a[2]
        
        y_hat = np.argmax(probs, axis=1)
        y_hat[y_hat != 0] = -1
        y_hat += 1
        
        # Create binary mask
        y_mask = y_hat.reshape((img.shape[0], img.shape[1]))
        y_mask = y_mask.astype(np.uint8)*255
        
        # Post-processing to remove noise
        kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT,(15,15))
        kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT,(9,9))
        mask1 = cv2.morphologyEx(y_mask, cv2.MORPH_OPEN, kernel2)
        mask2 = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE, kernel1)
        mask_img = y_mask * mask1
        mask_img = mask_img * mask2
        
        # Display original test image in RGB color space
#        x_test = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#        plt.imshow(x_test)
#        plt.show() 
        
        	# Display binary mask pre-processing
#        plt.imshow(y_mask, cmap='gray')
#        plt.show()
        # Display binary mask post-processing
#        plt.imshow(mask_img, cmap='gray')
#        plt.show()
        return mask_img
        
    def get_centroid(self, img):
        '''
            Find the centroid of the detected phone

			Inputs:
				img - original image
			Outputs:
				centroids - a list of candidate centroids of detected phones (x,y) 
				where (x, y) are normalized in respect to the size of the test image

		'''
        binary_img = self.segment_image(img)
        centroids = []
        h_img, w_img = binary_img.shape
        contours, hierarchy = cv2.findContours(binary_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key = cv2.contourArea, reverse = True) # reorder contours from large to small

        for contour in contours:
            if cv2.contourArea(contour) >= 250 and cv2.contourArea(contour) <= 700:
#                print(cv2.contourArea(contour))
                centroid, width_height, _ = cv2.minAreaRect(contour)
#                print('here', centroid)
                
                x, y = centroid[0]/w_img, centroid[1]/h_img # normalized centroid coords
                centroids.append([round(x,4), round(y,4)])
        
        if not centroids: # takes care of case of no detections from initial pass
            if len(contours) <= 0:
                centroids.append([0, 0])
            if len(contours) == 1: # if there's only 1 contour region, return this
                centroid, width_height, _ = cv2.minAreaRect(contours[0])
                x, y = centroid[0]/w_img, centroid[1]/h_img
                centroids.append([round(x,4), round(y,4)])
            else:
                for contour in contours:
                    if cv2.contourArea(contour) < 900:
#                        print(cv2.contourArea(contour))
                        centroid, width_height, _ = cv2.minAreaRect(contour)
#                        print('here2', centroid)
                        x, y = centroid[0]/w_img, centroid[1]/h_img
                        centroids.append([round(x,4), round(y,4)])
        return centroids[0]

if __name__ == '__main__':
    folder = './find_phone'
    fname = './find_phone/labels.txt'
    
    with open (fname, 'r') as myfile:
        img = []
        labels = []
        for line in myfile:
            words = line.split()
            img.append(words[0])
            labels.append((float(words[1]), float(words[2])))

    X_train = img
    end = len(X_train)
    
    # instantiate Phone Detector
    my_detector = PhoneDetector()

    for i in range(end):
        filename = X_train[i]
        img = cv2.imread(os.path.join(folder,filename))
        
        detect = my_detector.get_centroid(img)
        print('train image %d:' %i)
        print('phone detected at:', detect[0], detect[1])
        print('phone actually at:', labels[i][0], labels[i][1], '\n')