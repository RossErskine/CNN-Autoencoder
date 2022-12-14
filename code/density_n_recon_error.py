#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
density and reconstruction error for a threshold to classify outliers
Calculates the mean and standard deviation of a batch of images:
    Loops over the batch, 
    encodes the images, 
    flattens image, 
    calculates density/pdf
    calculates reconstruction error
    calculates and reurns mean and standard deviation
    
Arguments:
    encoder, decoder: CAE from CNN_Autoencoder.py
    batch_images fro torch dataloader constructed ImageGenerator.py
    kde from sklearn.neighbours KernelDensity
    Loss_fn from pytorch nn,Module
    
Example:
    calc_density_and_recon_error(encoder, decoder, batch_images, kde, loss_fn):
    

Created on Tue Dec  6 13:14:42 2022

@author: Ross Erskine

"""
import numpy as np


def calc_density_and_recon_error(encoder, decoder, batch_images, kde, loss_fn):
    """Calculates the mean and standard deviation of a batch of images:"""
    density_list=[]
    recon_error_list=[]
    
    for im in range(0, batch_images[1].size(dim=0)-1):
        
        # selct each image
        img  = batch_images[0][im,:,:,:]
        
        # Create a compressed version of the image using the encoder
        encoded_img = encoder(img) 
        
        # Flattens and adde another dimension to claculate density
        flattened_img = encoded_img.detach().flatten().numpy()
        flattened_img = flattened_img[np.newaxis, :]
           
        # get a density score for the new image
        density = kde.score_samples(flattened_img)[0] 
        
        # Calculate reconstruction error
        reconstruction = decoder(encoded_img)
        reconstruction_error = loss_fn(reconstruction, img)
        reconstruction_error = reconstruction_error.detach().numpy()
        
        # add to list
        density_list.append(density)
        recon_error_list.append(reconstruction_error)
      
    # calculate density mean and standard deviation
    average_density = np.mean(np.array(density_list))  
    stdev_density = np.std(np.array(density_list)) 
    
    # calculate reconstruction mean and standard deviation
    average_recon_error = np.mean(np.array(recon_error_list))  
    stdev_recon_error = np.std(np.array(recon_error_list)) 
    
    return average_density, stdev_density, average_recon_error, stdev_recon_error

