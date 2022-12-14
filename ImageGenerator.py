#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image generator
WARNING: NEED Parameter.py class
retruns training dataloader and validation dataloader
Example:
    parameters = Parameters()
    train_generator, validation_generator = train_dataloader(filename, parameters)

creates a torchvision Dataloader
Splits data ratio 8:2
Transforms data 128x128


Created on: Mon Nov 14 11:07:20 2022

Last Commited: 13/12/22

@author: Ross Erskine (ppxre1)
"""


import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader,random_split
import os
import Parameters as para

def train_dataLoader(filename = './test_images', param=para.Paramaters()):
    """ returns a pytorch data loader"""
    if os.path.exists(filename)==False:
        print('file not found!')
    
    dataset= torchvision.datasets.ImageFolder(filename)# gets dtaset
    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Resize(param.get_image_size())]) 
    dataset.transform = data_transform # transforms images using above data_transform
    m=len(dataset)# dataset size
    
    train_set_size = int(m*0.8) # ratio 8:2
    train_data, val_data = random_split(dataset, [train_set_size, int(m-train_set_size)])#splits data
    
    train_loader = DataLoader(train_data, batch_size=param.get_batch_size()) #creates dataloaders
    val_loader = DataLoader(val_data, batch_size=param.get_batch_size())
    
    return [train_loader, val_loader]
   
    
