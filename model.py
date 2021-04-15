import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import cv2
import os
import tensorflow as tf


key_points_df= pd.read_csv('data/training_frames_keypoints.csv')

class make_dataset(tf.keras.utils.Sequence):
    def __init__(self, csv_dir, root_dir, batch_size):
        self.key_points_frame= pd.read_csv(csv_dir)
        self.root_dir= root_dir
        self.batch_size = batch_size
        

    def __len__(self):
        return len(self.key_points_frame)
    
    def __getitem__(self, index):
        image_name= os.path.join(self.root_dir,self.key_points_frame.iloc[index,0])
        image = cv2.imread(image_name)
        image= image[:,:,0:3]
        
        h, w = image.shape[:2]
        
        image_resized = cv2.resize(image, (192,192))
        
        if (len(image_resized.shape)==2):
            image_resized = image_resized.reshape(image_resized.shape[0],image_resized.shape[1],1)
            
        
        
        key_points= self.key_points_frame.iloc[index, 1:].values
        key_points= key_points.astype('float').reshape(-1,2)
        
        key_points = key_points * [192/w, 192/h]

        sample ={'image':image_resized , 'key_points': key_points}
        
        return sample
    

sample= make_dataset("data/training_frames_keypoints.csv", "data/training/", 32)
    
X = np.asarray([sample['Image']], dtype=np.uint8)


                               
print("Number of images are", len(x_train))

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Conv2D(filters =32, kernel_size =3,activation = 'relu', input_shape=[192, 192, 3] ))
model.add(tf.keras.layers.MaxPool2D(pool_size =2))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Conv2D(filters =64, kernel_size= 3, activation ='relu'))
model.add(tf.keras.layers.MaxPool2D(pool_size=2))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(units=128, activation='relu'))
model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

model.fit(x = x_train, validation_data = x_test, epochs = 25)   
   