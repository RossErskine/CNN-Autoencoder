#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 09:31:36 2022

Main is the interface for which the project CNN Autoencoder will start

@author: Ross Erskine
"""

from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os
from PIL import Image
import Parameters as para

filename = './test_images'
param = para.Paramaters(batch_size=32)

datagen = ImageDataGenerator()

train_gen = datagen.flow_from_directory(
    filename,
    target_size=(param.get_image_size()),
    batch_size=param.get_batch_size())

for i in range(5):
    img, label = train_gen.next()
    plt.figure(figsize=(6,6))
    plt.imshow(train_gen[0][i])
    plt.show()