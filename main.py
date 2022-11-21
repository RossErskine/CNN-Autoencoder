#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 09:31:36 2022

Main is the interface for which the project CNN Autoencoder will start

@author: Ross Erskine
"""


import matplotlib.pyplot as plt
import random
import tensorflow as tf
import torch
import numpy as np

# import modules
import Parameters as pr
import ImageGenerator as ig
from CNN_Autoencoder import Encoder, Decoder


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


############################## Constructors ##################################
params = pr.Paramaters() # Construct Paramaters
MSE_los = torch.nn.MSELoss()
lr = 0.001

train_loader, val_loader = ig.train_dataLoader()

encoder = Encoder()
decoder = Decoder()
params_to_optimize = [
    {'params': encoder.parameters()},
    {'params': decoder.parameters()}
]
optim = torch.optim.Adam(params_to_optimize, lr=lr, weight_decay=1e-05)

########################### GPU #############################################

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Selected device: {device}')

# Move both the encoder and the decoder to the selected device
encoder.to(device)
decoder.to(device)

########################### Training functions ################################
def train_epoch(encoder, decoder, device, dataloader, loss_fn, optimizer):
    # Set train mode for both the encoder and the decoder
    encoder.train()
    decoder.train()
    train_loss = []
    # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
    for image_batch, _ in dataloader: # with "_" we just ignore the labels (the second element of the dataloader tuple)
        # Move tensor to the proper device
        image_batch = image_batch.to(device)
        # Encode data
        encoded_data = encoder(image_batch)
        # Decode data
        decoded_data = decoder(encoded_data)
        # Evaluate loss
        loss = loss_fn(decoded_data, image_batch)
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Print batch loss
        print('\t partial train loss (single batch): %f' % (loss.data))
        train_loss.append(loss.detach().cpu().numpy())

    return np.mean(train_loss)

########################### Training ########################################
num_epochs = 30
diz_loss = {'train_loss':[],'val_loss':[]}
for epoch in range(num_epochs):
   train_loss =train_epoch(encoder,decoder,device,
   train_loader,MSE_los,optim)
   #val_loss = test_epoch(encoder,decoder,device,test_loader,loss_fn)
   print('\n EPOCH {}/{} \t train loss {}'.format(epoch + 1, num_epochs,train_loss))
   diz_loss['train_loss'].append(train_loss)
   #diz_loss['val_loss'].append(val_loss)
   #plot_ae_outputs(encoder,decoder,n=10)