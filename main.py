#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 09:31:36 2022

Main is the interface for which the project CNN Autoencoder will start

Source: https://medium.com/dataseries/convolutional-autoencoder-in-pytorch-on-mnist-dataset-d65145c132ac

@author: Ross Erskine
"""


import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
import numpy as np

# import modules
import Parameters as pr
import ImageGenerator as ig
import CNN_Autoencoder as CAE

############################## Constructors ##################################
params = pr.Paramaters() # Construct Paramaters

traingen, valgen = ig.train_dataLoader()  # Construct ImageGenerator

loss_fn = torch.nn.MSELoss()


model = CAE.Convolutional_Autoencoder()     # Construct model

optimizer = torch.optim.Adam(model.parameters(),
                             lr=1e-3, 
                             weight_decay=1e-5)

########################### GPU check ########################################
# Check if the GPU is available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#print(f'Selected device: {device}')

# Move both the encoder and the decoder to the selected device
#model.to(device)


########################### Train function ########################################
def train_model(model, dataloader, device, loss_fn, optimizer):
    
    train_loss = []
    for (img, _) in traingen:
        # move tensor to device
        #mg = img.to(device)
        # 
        recon = model(img)
        loss = loss_fn(recon, img)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print('\t partial train loss (single batch): %f' % (loss.data))
    train_loss.append(loss.detach().cpu().numpy())
    
    return np.mean(train_loss)

########################### Train and evaluate ########################################
num_epochs = 30
diz_loss = {'train_loss':[],'val_loss':[]}
for epoch in range(num_epochs):
   train_loss =train_model(model,device,
   traingen,loss_fn,optimizer)
   #val_loss = test_epoch(encoder,decoder,device,test_loader,loss_fn)
   print('\n EPOCH {}/{} \t train loss {}'.format(epoch + 1, num_epochs,train_loss))
   diz_loss['train_loss'].append(train_loss)
   #diz_loss['val_loss'].append(val_loss)
   #plot_ae_outputs(model,n=10)