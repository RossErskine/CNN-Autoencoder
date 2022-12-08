#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Encode Samples

Created on Tue Dec  6 12:51:23 2022

@author: Ross Erskine
"""
import pandas as pd
import torch
from tqdm import tqdm

def encode_samples(encoder, dataloader):
    """ Encodes Samples """
    
    encoded_batches = []
    for batch, _ in tqdm(dataloader):
        
        encoded_batch = encoder(batch)
        encoded_batch = [img.detach().flatten().numpy() for img in encoded_batch]
        
        # Append to list
        encoded_batches.append(encoded_batch)
    encoded_batches = pd.DataFrame(encoded_batch)
    encoded_batches = encoded_batches.dropna()
    return encoded_batches