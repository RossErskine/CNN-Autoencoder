#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
create training data

Original dataset has 243,437 images. 
This script is to lower the dataset size to around 20,000

Created on Wed Nov 23 18:39:44 2022
Last commited: 13/12/22

@author: Ross Erskine ppxre1
"""

import os
import shutil

def create_training_data(source= '../datasets/galaxy_images',
                         destination = '../datasets/galaxy_training_imgs'):
    """Moves a specified amount of objects from on file to another"""
    
    allfiles = os.listdir(source)
    
    for i in range(20000):
        src_path = os.path.join(source, allfiles[i])
        dst_path = os.path.join(destination, allfiles[i])
        shutil.move(src_path, dst_path)
     

        
        