#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test class for ImageGenerator.py 
Tests default file path 

Created on Tue Dec 13 19:42:50 2022

@author: roscopikotrain
"""

import unittest
import os

class TestImageGenerator(unittest.TestCase):
    """ Test ImageGenerator class """
    
    def testFilePath(self):
        """Test default file path"""
        filename = './test_images'
        msg = "File path is not True"
        self.assertTrue(os.path.exists(filename), msg)
        
if __name__ == '__main__': 
          
    unittest.main()