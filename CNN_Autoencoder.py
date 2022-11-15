#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CNN Autoencoder Class


Created on Mon Nov 14 09:39:52 2022
Tested: TODO
Last commited: 


@author: Ross Erskine ppxre1
"""
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D

SIZE = 128
def CAE_model():
    #Encoder
    model = Sequential()
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(SIZE, SIZE, 3)))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), padding='same'))
    
    #Decoder
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    
    model.add(Conv2D(3, (3, 3), activation='sigmoid', padding='same'))
    
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])
    
    return model

if __name__ == '__main__': 
    
############################# Testing #####################################
    
    """
    Test class for CNN_Autoencoder
    """
    import unittest     
    
    class Test_CNN_Autoencoder(unittest.TestCase):
        """Test CNN Autoencoder"""
        
        def test_constructor(self):
            """Test CNN Autoencoder constructor"""
            CAE = CAE_model()
            CAE.summary()
            
            
            
    unittest.main()   