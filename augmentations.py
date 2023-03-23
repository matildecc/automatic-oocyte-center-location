# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 07:57:51 2023

Augmentations
@author: Matilde
"""
#%% Flip horizontal

import cv2
import os

source_path =  "./database - immature/original images/"
save_pathH = "./database - immature/hflipped/"
save_pathV = "./database - immature/vflipped/"
save_pathVH = "./database - immature/vhflipped/"

for path in os.listdir(source_path):   
    image= cv2.imread(source_path+path)
    flippedimageH= cv2.flip(image, 1)
    cv2.imwrite(save_pathH + path, flippedimageH)
    
    #labels 
    # (2592 - x), y
    
    flippedimageV = cv2.flip(image, 0)
    cv2.imwrite(save_pathV + path, flippedimageV)
    
    #labels 
    # (x, 1944 - y)

    flippedimageVH = cv2.flip(image, -1)
    cv2.imwrite(save_pathVH + path, flippedimageVH)
    
    #labels 
    # (2592 - x, 1944 - y)
