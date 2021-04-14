import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import cv2
import os

key_points_df= pd.read_csv('data/training_frames_keypoints.csv')

class make_dataset:
    def __init__(self, csv_dir, root_dir):
        self.key_points = pd.read_csv(csv_dir)
        self.root_dir= root_dir

    def __len__(self):
        return len(self.key_points)
    
    def __getitem__(self, index):
        image_name= os.path.join(self.root_dir,self.key_points.iloc[index,0])
        image = cv2.imread(image_name)
        image= image[:,:,0:3]
        
        h, w = image.shape[:2]
        
        image_resized = cv2.resize(image, (192,192))
        
        
        key_points= self.key_points.iloc[index, 1:].values
        key_points= key_points.astype('float').reshape(-1,2)
        
        key_points = key_points * [192/w, 192/h]

        sample ={'image':image_resized , 'key_points': key_points}

        return sample
    
x_train = make_dataset(csv_dir='data/training_frames_keypoints.csv', 
                           root_dir= 'data/training/')
print("Number of images are", len(x_train))

sample = x_train[5]
kp= sample['key_points']
print(sample['image'].shape, sample['key_points'].shape)
plt.imshow(sample['image'])
plt.scatter(kp[:, 0], kp[:, 1], s=20, marker='.', c='m')
    