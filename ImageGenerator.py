#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image generator



Created on: Mon Nov 14 11:07:20 2022
Tested: TODO
Last Commited: 


@author: Ross Erskine (ppxre1)
"""
from keras.preprocessing.image import ImageDataGenerator
import os
# from PIL import Image
import Parameters as para

def train_imageGenerator():
    """ returns a training generator"""
    
    filename = '/test_image/galaxy'
    param = para.Paramaters()
        
    datagen = ImageDataGenerator()

    train_gen = datagen.flow_from_directory(
    filename,
    target_size=(param.get_image_size),
    batch_size=param.get_batch_size)
    return train_gen
    
    
        
    
    
    
    
if __name__ == '__main__': 
    
########################## Testing ##########################################

    import unittest
    
    class TestImageGenerator(unittest.TestCase):
        """ Test ImageGenerator class """
        
        def testFilePath(self):
            """Test file path"""
            filename = './test_images'
            msg = "File path is not True"
            self.assertTrue(os.path.exists(filename), msg)
            
        
            
    unittest.main()