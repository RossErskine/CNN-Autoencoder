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
    
    encoded_samples = []
    for sample in tqdm(dataloader):
        img = sample[0]
        #label = sample[1]
        # Encode image
        encoder.eval()
        with torch.no_grad():
            encoded_img  = encoder(img)
        # Append to list
        encoded_img = encoded_img.flatten().cpu().numpy()
        encoded_sample = {f"Enc. Variable {i}": enc for i, enc in enumerate(encoded_img)}
        #encoded_sample['label'] = label
        encoded_samples.append(encoded_sample)
    encoded_samples = pd.DataFrame(encoded_samples)
    encoded_samples = encoded_samples.dropna()
    return encoded_samples