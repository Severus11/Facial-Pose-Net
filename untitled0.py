# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 14:55:52 2021

@author: parth
"""


import tensorflow as tf
if tf.test.gpu_device_name():
    print('Default GPU Device:{}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")