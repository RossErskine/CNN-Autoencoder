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


class CNN_Autoencoder(tf.keras.Model):
    def __init__(self, size = 128):
        super().__init__()
        
        self._size = size
        
        self.encoder = tf.keras.Sequential()
        self.encoder.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(self._size, self._size, 3)))
        self.encoder.add(MaxPooling2D((2, 2), padding='same'))
        self.encoder.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
        self.encoder.add(MaxPooling2D((2, 2), padding='same'))
        self.encoder.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
        self.encoder.add(MaxPooling2D((2, 2), padding='same'))
        
        self.decoder = tf.keras.Sequential()
        self.decoder.add(Conv2D(16, (3, 3), activation='relu', padding='same',input_shape=(16, 16, 3)))
        self.decoder.add(UpSampling2D((2, 2)))
        self.decoder.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
        self.decoder.add(UpSampling2D((2, 2)))
        self.decoder.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        self.decoder.add(UpSampling2D((2, 2)))

        self.decoder.add(Conv2D(3, (3, 3), activation='sigmoid', padding='same'))
        self.decoder.build(input_shape=(16, 16))
        
    def call(self, X):
        latent_X = self.encoder(X) # compress
        decoded_X = self.decoder(latent_X) # unpack
        
        return decoded_X
        

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
            CNNAE = CNN_Autoencoder()
            CNNAE.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])
            
            
    unittest.main()   