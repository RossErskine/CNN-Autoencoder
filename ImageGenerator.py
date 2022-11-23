#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image generator



Created on: Mon Nov 14 11:07:20 2022
Tested: TODO
Last Commited: 


@author: Ross Erskine (ppxre1)
"""

import torch.utils.data as data
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader,random_split
import os
# from PIL import Image
import Parameters as para

def train_dataLoader(filename = './test_images'):
    """ returns a training generator"""
    
    #filename = './test_images'
    #filename = '../datasets'
    param = para.Paramaters()
        
    dataset= torchvision.datasets.ImageFolder(filename)
    data_transform = transforms.Compose([transforms.ToTensor(),transforms.Resize([128,128])])
    dataset.transform = data_transform
    m=len(dataset)
    
    train_set_size = int(m*0.8)
    train_data, val_data = random_split(dataset, [train_set_size, int(m-train_set_size)])
    
    train_loader = DataLoader(train_data, batch_size=param.get_batch_size())
    val_loader = DataLoader(val_data, batch_size=param.get_batch_size())
    
    return [train_loader, val_loader]
    
    
def get_test_set():
    """reurns a test set of images rather than a dataloader"""
    filename = './test_images'
    test_set= torchvision.datasets.ImageFolder(filename)
    data_transform = transforms.Compose([transforms.ToTensor(),transforms.Resize([128,128])])
    test_set.transform = data_transform
    return test_set
    
    
    
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
            
        def Test_dataloader_constructor(self):
            """Test dataloader constructor """
            data_loader= train_dataLoader()
            
        
            
    unittest.main()