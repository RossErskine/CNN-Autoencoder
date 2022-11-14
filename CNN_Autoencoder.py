#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 09:39:52 2022

Class CNN_Autoencoder

@author: Ross Erskine ppxre1
"""
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D

class CNN_Autoencoder(tf.keras.Model):
    def __init__(self):
        super()._init__()
        
        self.encoder = tf.keras.Sequential([
            # TODO Compress into small dimension
            ])
        
        self.decoder = tf.keras.Sequential([
            # TODO unpack into original dimension
            ])
        
    def call(self, X):
        latent_X = self.encoder(X) # compress
        decoded_X = self.decoder(X) # unpack
        
        return decoded_X


if __name__ == '__main__': 
    
    """
    Test class for CNN_Autoencoder
    """
    import unittest      