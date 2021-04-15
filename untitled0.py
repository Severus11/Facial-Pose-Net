# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 12:21:07 2021

@author: parth
"""


import cv2
import matplotlib.pyplot as plt
import pandas as pd
import os

key_points_df = pd.read_csv('data/training_frames_keypoints.csv')


n= int(input('enter a random number bw 0 to 3464:'))

img_name = key_points_df.iloc[n, 0]
key_points = key_points_df.iloc[n, 1:].values
key_points = key_points.astype('float').reshape(-1,2)

image = cv2.imread(os.path.join('data/training/', img_name))
h, w = image.shape[:2]
image_resized = cv2.resize(image, (192,192))

key_points = key_points * [192 / w, 192 / h]
        
'''
plt.imshow(image_resized)
plt.scatter(key_points[:, 0], key_points[:, 1], s=20, marker='.', c='m')
print('img printed')'''




