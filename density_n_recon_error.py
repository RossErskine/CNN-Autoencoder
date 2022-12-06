#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate density and reconstruction error

Created on Tue Dec  6 13:14:42 2022

@author: Ross Erskine

"""
import numpy as np


#Calculate density and reconstruction error to find their means values for
#good and anomaly images. 
#We use these mean and sigma to set thresholds. 
def calc_density_and_recon_error(encoder, decoder, batch_images, kde, loss_fn):
    
    density_list=[]
    recon_error_list=[]
    for im in range(0, batch_images[1].size(dim=0)-1):
        
        img  = batch_images[im]
        #img = img[np.newaxis, :,:,:]
        
        
        encoded_img = encoder(img) # Create a compressed version of the image using the encoder
        #encoded_img = encoded_img.flatten().cpu().numpy()# Flatten the compressed image
        
        encoded_img = encoded_img.detach().numpy()
        
        
        density = kde.score_samples(encoded_img)[0] # get a density score for the new image
        reconstruction = decoder([[img]])
        reconstruction_error = loss_fn(reconstruction, img)
        density_list.append(density)
        recon_error_list.append(reconstruction_error)
        
    average_density = np.mean(np.array(density_list))  
    stdev_density = np.std(np.array(density_list)) 
    
    average_recon_error = np.mean(np.array(recon_error_list))  
    stdev_recon_error = np.std(np.array(recon_error_list)) 
    
    return average_density, stdev_density, average_recon_error, stdev_recon_error

