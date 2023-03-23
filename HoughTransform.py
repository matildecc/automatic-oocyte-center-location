# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 15:59:51 2023

@author: Matilde
"""


import sys
import os
import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

source_path =  "./database - immature/center cropped images/"

df = pd.read_excel('./database - immature/images_info.xlsx', dtype=str)


for path in os.listdir(source_path):   

     oocyte = (path.rpartition('-')[2][:-4])
     center_str = path.rpartition(' -')[0][9:]
     x = int(center_str.partition(',')[0])
     y = int(center_str.partition(',')[2])
     df.loc[df['oocyte n°'] == oocyte,'center x'] = x
     df.loc[df['oocyte n°'] == oocyte,'center y'] = y
     if not pd.isna((path)):
         df.loc[df['oocyte n°'] == oocyte,'image_name'] = path

        
new_df = df[(df["Accepted_AR (Y/N)"] == 'Y')]
#%%
center_images = []
error = []
for path in os.listdir("./database - immature/original images/"):
       
    oocyte = (path.rpartition('-')[2][:-4])
    
    if not (df[df['oocyte n°'] == oocyte].empty):
        
        x = int(df[df['oocyte n°'] == oocyte]['center x'])
        y = int(df[df['oocyte n°'] == oocyte]['center y'])
            
        filename = "./database - immature/original images/" + path
        # Loads an image
        src = cv.imread(cv.samples.findFile(filename), cv.IMREAD_COLOR)
        
        # Convert it to gray
        gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
        
        
        # Reduce the noise to avoid false circle detection
        gray = (cv.medianBlur(gray, 5)-255)*255
        
        
        rows = gray.shape[0]
        circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, rows / 8,
                                   param1=10, param2=30,
                                   minRadius=50, maxRadius=70)
        
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                center = (i[0], i[1])
                error.append([path, np.sqrt((x-i[0])**2 + (y-i[1])**2)])
                
                # circle center
                cv.circle(src, center, 1, (0, 100, 100), 3)
                # circle outline
                radius = i[2]
                cv.circle(src, center, radius, (255, 0, 255), 3)
                center_images.append([path[:-4], center])
        
        
        # cv.namedWindow("detected circles", cv.WINDOW_NORMAL)
        # cv.imshow("detected circles", src)
        # cv.waitKey(0)
    

#%% Selection of the closest circle 
corrected_error = []
error = np.array(error)

for elemt in error[:,0]:
    circle_min = min(error[error[:,0] == elemt, 1])
    
    corrected_error.append([elemt, circle_min])
    

corrected_error = np.array(corrected_error)    
corrected_error = np.unique(corrected_error, axis = 0)
    


#%% average error 

print( 'Average results', np.mean(corrected_error[:,1].astype(float)))
print('STD', np.std(corrected_error[:,1].astype(float)))