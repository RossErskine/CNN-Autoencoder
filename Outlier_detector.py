#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Outlier detection
using the already created pdf that will create a threshold to determine 
if each image is an oultier.
Loops through a bacth of images:
    sets threshold,
    encodes image,
    flattens encoded image
    calculates density
    calculates reconstruction error
    classifiies image 
    returns result

Arguments:
    encoder, decoder: CAE from CNN_Autoencoder.py
    batch_images fro torch dataloader constructed ImageGenerator.py
    kde from sklearn.neighbours KernelDensity
    pdf creates from density_n_recon_error.py
    Loss_fn from pytorch nn,Module
    
Example:
    check_outlier(img, encoder, decoder, kde, pdf, loss_fn)
    
Created on Mon Dec 12 12:18:50 2022

@author: roscopikotrain
"""

import numpy as np

def check_outlier(img, encoder, decoder, kde, pdf, loss_fn):
    """using the already created pdf that will create a threshold to determine 
        if each image is an oultier.
    """
    
    #sets threshold
    density_threshold = pdf[1] #pdf
    reconstruction_error_threshold = pdf[3] # Set this value based on the above exercise
    # encodeds image
    encoded_img = encoder(img)
    #flattens image
    flattened_img = encoded_img.detach().flatten().numpy()
    flattened_img = flattened_img[np.newaxis, :]
    # calculates image
    density = kde.score_samples(flattened_img)[0] # get a density score for the new image
    # claculates reconstruction error 
    reconstruction = decoder(encoded_img)
    reconstruction_error = loss_fn(reconstruction, img)
    reconstruction_error = reconstruction_error.detach().numpy()
    
    ## classifys image and returns output
    if density > density_threshold or reconstruction_error < reconstruction_error_threshold:
        
        return 1
    else:
        
        return 0