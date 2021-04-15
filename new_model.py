# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 01:26:13 2021

@author: parth
"""


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import cv2
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split 

key_points_df= pd.read_csv('data/training_frames_keypoints.csv')

def make_dataset(csv_dir, root_dir):
    
    key_points_frame = pd.read_csv(csv_dir)
    x=[]
    y=[]
    
    for i in range(len(key_points_frame)):
        image_name= os.path.join(root_dir,key_points_frame.iloc[i,0])
        image = cv2.imread(image_name)
        image= image[:,:,0:3]
        #image= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = image.shape[:2]
        image_resized = cv2.resize(image, (192,192))
        x.append(image_resized)
        
        key_points= key_points_frame.iloc[i, 1:].values
        key_points= key_points.astype('float').reshape(-1,2)
        key_points = key_points * [192/w, 192/h]
        key_points= key_points.reshape(-1)
        y.append(key_points)
        
    x = np.asarray(x, dtype=np.float32)
    y= np.asarray(y, dtype=np.float32)
    return x, y
   
x_train, y_train = make_dataset("data/training_frames_keypoints.csv", "data/training/")

X_train, X_test, Y_train, Y_test = train_test_split(x_train, y_train, test_size=0.3, random_state=42)


model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Conv2D(filters =32, kernel_size =(3,3), padding='same', input_shape=(192, 192, 3)))
model.add(tf.keras.layers.LeakyReLU(alpha = 0.1))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(filters =32, kernel_size =(3,3),padding='same'))
model.add(tf.keras.layers.LeakyReLU(alpha = 0.1))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPool2D(pool_size =2))

model.add(tf.keras.layers.Conv2D(filters =64, kernel_size= (3,3),padding='same'))
model.add(tf.keras.layers.LeakyReLU(alpha = 0.1))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(filters =64, kernel_size= (3,3),padding='same'))
model.add(tf.keras.layers.LeakyReLU(alpha = 0.1))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPool2D(pool_size=2))

model.add(tf.keras.layers.Conv2D(filters =128, kernel_size= (3,3),padding='same'))
model.add(tf.keras.layers.LeakyReLU(alpha = 0.1))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(filters =128, kernel_size= (3,3),padding='same'))
model.add(tf.keras.layers.LeakyReLU(alpha = 0.1))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPool2D(pool_size=2))

model.add(tf.keras.layers.Conv2D(filters =256, kernel_size= (3,3),padding='same'))
model.add(tf.keras.layers.LeakyReLU(alpha = 0.1))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(filters =256, kernel_size= (3,3),padding='same'))
model.add(tf.keras.layers.LeakyReLU(alpha = 0.1))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPool2D(pool_size=2))

model.add(tf.keras.layers.Conv2D(filters =512, kernel_size= (3,3),padding='same'))
model.add(tf.keras.layers.LeakyReLU(alpha = 0.1))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(filters =512, kernel_size= (3,3),padding='same'))
model.add(tf.keras.layers.LeakyReLU(alpha = 0.1))
model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(units=512, activation='relu'))
model.add(tf.keras.layers.Dropout(0.1))
model.add(tf.keras.layers.Dense(units=136))
model.compile(optimizer = 'adam', loss = 'mse', metrics = ['accuracy', 'mae'])

checkpoint = tf.keras.callbacks.ModelCheckpoint('model_weights.h5', monitor=['val_accuracy'],save_weights_only=True, mode='max', verbose=1)
#reduce_lr= tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor = 0.1, patience=2, min_delta=0.00001, mode='auto')
#tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")
#callbacks = [tensorboard_callback,checkpoint, reduce_lr]

model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs = 500)  

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])