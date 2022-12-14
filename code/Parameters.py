#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parameters Class

Paramaters Class holds pararmeters for model CNN_autoencoder

Example:
foo = Parameters(batch_size= int, img_height = int, img_width = int)

defaults are:
    batch size = 32
    image height = 128
    image width = 128
    
Tests:
    Default
    constructor

Created on: Mon Nov 14 11:15 2022
Tested: Mon Nov 14 12:10 2022
Last Commited: Mon Nov 14 12:15 2022

@author: Ross Erskine (ppxre1)
"""

class Paramaters:
    def __init__(self, batch_size = 32, img_height = 128, img_width = 128):
        """Parameter constructor """
        self._batch_size = batch_size
        self._img_height = img_height
        self._img_width = img_width
        
    def get_batch_size(self):
        """return batch size"""
        return self._batch_size
    
    def get_image_size(self):
        """return image size in array"""
        return [self._img_height, self._img_width]
    

    
    