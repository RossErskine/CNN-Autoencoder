#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Encode Samples

Uses the Encoder class from CNN_Autoencoder.py to reduce dimensionality 
encoded images.loops through each batch of a pytorh dataloader encoding each image.
if some batches are not full encoder retruns nan values. Pandas dataframe used to eliminate these images.
    uses 'tqdm' which is a progress bar
    
    Example:
        encode_samples(encoder, dataloader)
    
     

Created on Tue Dec  6 12:51:23 2022

@author: Ross Erskine
"""
import pandas as pd
from tqdm import tqdm

def encode_samples(encoder, dataloader):
    """ Encodes Samples: loops through each batch in dataloader encoding each image"""
    
    encoded_batches = []
    for batch, _ in tqdm(dataloader):
        # enodes images
        encoded_batch = encoder(batch)
        # flattens images in each batch
        encoded_batch = [img.detach().flatten().numpy() for img in encoded_batch]
        # Append to list
        encoded_batches.append(encoded_batch)
    # pandas to drop nan objects
    encoded_batches = pd.DataFrame(encoded_batch)
    encoded_batches = encoded_batches.dropna()
    encoded_batches = encoded_batches.to_numpy()
    return encoded_batches