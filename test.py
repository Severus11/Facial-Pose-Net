import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import cv2
import os

key_points_df= pd.read_csv('data/training_frames_keypoints.csv')

n= int(input('enter a random number bw 0 to 3464:'))

img_name = key_points_df.iloc[n, 0]
key_points = key_points_df.iloc[n, 1:].values
key_points = key_points.astype('float').reshape(-1,2)

img = mpimg.imread(os.path.join('data/training/', img_name))
plt.imshow(img)
plt.scatter(key_points[:, 0], key_points[:, 1], s=20, marker='.', c='m')
print('img printed')