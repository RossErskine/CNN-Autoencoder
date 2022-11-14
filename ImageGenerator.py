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
import matplotlib.pyplot as plt
import os
from PIL import Image
import Parameters as para

class imageGenerator(ImageDataGenerator):
    def __init__(self, filename, param = para.Paramaters()):
        super().__init__()
        self._filename = filename
        self._param = param
        
    def train_generator(self):
        """ returns a training generator"""
        datagen = ImageDataGenerator()

        train_gen = datagen.flow_from_directory(
            self._filename,
            target_size=(self._param.get_image_size),
            batch_size=self._param.get_batch_size)
        return train_gen
    
    def _next(self):
        """returns ImageDataGenerator.next()"""
        datagen = train_generator()
        return 
        
    
    
    
    
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
            
        def testImageGenerator(self):
            """Test image Generator"""
            filename = './test_images'
            param = para.Paramaters()
            test_imgGen = imageGenerator(filename, param)
            for _ in range(5):
                img, label = test_imgGen.next()
                print(img.shape)   #  (1,256,256,3)
                plt.imshow(img[0])
                plt.show()
            
    unittest.main()