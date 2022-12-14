#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
helper function
reshapes image
Created on Wed Dec 14 10:30:52 2022

@author: roscopikotrain
"""
import matplotlib.pyplot as plt
import numpy as np


def imshow(img):
    # helper function to reshape and display an image
    img = img / 2 + 0.5  # normalize
    plt.imshow(np.transpose(img, (1, 2, 0)))  # convert from Tensor image
