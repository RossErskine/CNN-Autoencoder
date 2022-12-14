#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Parameters class
Test constructors and defaults
Created on Wed Dec 14 13:17:03 2022

@author: roscopikotrain
"""

import unittest
import Parameters as params
    
class TestParameters(unittest.TestCase):
    """ Test Parameter class """
    
    def test_Pararmeters_default(self):
        """ test Parameter class defaults"""
        testParameters = params.Paramaters()
        self.assertEqual(testParameters.get_batch_size(),32)
        self.assertEqual(testParameters.get_image_size(), [128,128 ])
        
    def test_Parameters_costructor(self):
        """test Parameter class constructor"""
        testParameters = params.Paramaters(batch_size=64, img_height=256, img_width=256)
        self.assertEqual(testParameters.get_batch_size(),64)
        self.assertEqual(testParameters.get_image_size(), [256, 256])
        
if __name__ == '__main__': 
          
    unittest.main()