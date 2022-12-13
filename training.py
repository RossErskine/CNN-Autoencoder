#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training funtions
two functions for training:
    train_model updates the weights returns mean training loss every epoch
    arguments:
        encoder: Encoder class CNN_Autoencoder.py
        decoder: Decoder class CNN_Autoencoder.py
        dataloader: ImageGenerator.py
        device: i.e. 'CPU or GPU'
        loss_fn: i.e. Mean sqaured error
        Optimiser: i.e. adam
        
    val_model returns mean generalaisation error every epoch
    arguments:
        encoder: Encoder class CNN_Autoencoder.py
        decoder: Decoder class CNN_Autoencoder.py
        dataloader: ImageGenerator.py
        device: i.e. 'CPU or GPU'
        loss_fn: i.e. Mean sqaured error

Created on Mon Dec  5 14:09:51 2022
last commited: 13/12/22

@author: Ross Erskine ppxre1
"""

import numpy as np
import torch

########################### Train function ###################################
def train_model(encoder, decoder, dataloader, device, loss_fn, optimizer):
    """ Train model using dataloader return the mean training loss """
    
    # Set evaluation mode for encoder and decoder
    encoder.train()
    decoder.train()
    
    train_loss = []
    for (img, _) in dataloader:
        # move tensor to device
        # NOT WORKING!
        #img = img.to(device)
        
        #Reconstruction error
        encoded_data = encoder(img)
        recon = decoder(encoded_data) 
        loss = loss_fn(recon, img)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    train_loss.append(loss.detach().cpu().numpy())
    
    return np.mean(train_loss)

########################### Validation function ##############################

def val_model(encoder, decoder,  dataloader, device,loss_fn):
    """ Validate model using dataloader return the mean Validation loss """
    # Set evaluation mode for encoder and decoder
    encoder.eval()
    decoder.eval()
    with torch.no_grad(): # No need to track the gradients
        
        val_loss = []
        for (img, _) in dataloader:
            # Move tensor to the proper device
            # NOT WORKING!
            #image_batch = image_batch.to(device)
            
            #Reconstruction error
            encoded_data = encoder(img)
            recon = decoder(encoded_data) 
            loss = loss_fn(recon, img)
            
        val_loss.append(loss.detach().cpu().numpy())
            
    return np.mean(val_loss)