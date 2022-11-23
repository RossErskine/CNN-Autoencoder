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
params = pr.Paramaters(batch_size=512) # Construct Paramaters

test_images = ig.get_test_set()
filename = '../datasets'
traingen, valgen = ig.train_dataLoader(filename)  # Construct ImageGenerator

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
    """ Train model using dataloader return the mean training loss """
    
    #model.train()
    train_loss = []
    for (img, _) in traingen:
        # move tensor to device
        #mg = img.to(device)
        
        #Reconstruction error
        recon = model(img)
        loss = loss_fn(recon, img)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print('\t partial train loss (single batch): %f' % (loss.data))
    train_loss.append(loss.detach().cpu().numpy())
    
    return np.mean(train_loss)

##################### helper function ###########################
def imshow(img):
    # helper function to un-normalize and display an image
    img = img / 2 + 0.5  # unnormalize
    plt.imshow(np.transpose(img, (1, 2, 0)))  # convert from Tensor image
    

########################### plot images function ########################################
def plot_outputs(model):
    """Plots after every epoch"""
    #obtain one batch of test images
    dataiter = iter(valgen)
    images, _ = next(dataiter)
    
    # get sample outputs
    output = model(images)
    # prep images for display
    images = images.numpy()
    
    
    # output is resized into a batch of iages
    #output = output.view(params.get_batch_size(), 3, 128, 128)
    # use detach when it's an output that requires_grad
    output = output.detach().numpy()
    
    # plot the first ten input images and then reconstructed images
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True, figsize=(24,4))
    for idx in np.arange(10):
        ax = fig.add_subplot(2, 10/2, idx+1, xticks=[], yticks=[])
        imshow(output[idx])
        ax.set_title("Reconstructed images")  
        
    # plot the first ten input images and then reconstructed images
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True, figsize=(24,4))
    for idx in np.arange(10):
        ax = fig.add_subplot(2, 10/2, idx+1, xticks=[], yticks=[])
        imshow(images[idx])
        ax.set_title("Original images")  

########################### Train and evaluate ########################################
n_of_iters = 100
num_epochs = 30
diz_loss = {'train_loss':[]}
for i in range(n_of_iters):
    for epoch in range(num_epochs):
       train_loss =train_model(model,device, traingen,loss_fn,optimizer)
       
       print('\n EPOCH {}/{} \t train loss {}'.format(epoch + 1, num_epochs,train_loss))
       diz_loss['train_loss'].append(train_loss)
       
    plot_outputs(model)