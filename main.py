#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 09:31:36 2022

Main is the interface for which the project CNN Autoencoder will run. 
Example: 
    runfile('main.py')
    
    
Sections:
    Imports:
        Libraries and modules
    Constructors:
        Parameter class, dataloader
    GPU check:
        check if GPU device is working
    Plot Images function:
        Plots the images from training on the last epoch of images before 
        and after being passed through CAE
    Train and evalute:
        train and validate the model using functions:
            train_model()
            val_model()
    Save and load model:
        save model
    KDE:
        Creates the PDF from the kernel density function KDE
    Outlier detection:
        checks for outliers from a a batch of validation data
    Final scatter:
        Plots validation batch data
    


Source: https://medium.com/dataseries/convolutional-autoencoder-in-pytorch-on-mnist-dataset-d65145c132ac
        https://towardsdatascience.com/introduction-to-autoencoders-b6fc3141f072
        https://github.com/bnsreenu/python_for_microscopists/tree/master/260_image_anomaly_detection_using_autoencoders
        https://github.com/patrickloeber/pytorch-examples
        
        

@author: Ross Erskine ppxre1
"""
####################### Imports ############################################

import matplotlib.pyplot as plt
import torch
from sklearn.neighbors import KernelDensity
import numpy as np


# import modules
import Parameters as pr
import ImageGenerator as ig
import CNN_Autoencoder as CAE
from training import train_model, val_model
from encode_samples import encode_samples
from density_n_recon_error import calc_density_and_recon_error
from Outlier_detector import check_outlier
from helper_function import imshow


############################## Constructors ##################################
# Construct Paramaters
params = pr.Paramaters(batch_size=64) 

# File path
filename = '../training' 
# Construct ImageGenerator
traingen, valgen  = ig.train_dataLoader(filename, params)  
# Mean-square error loss function
loss_fn = torch.nn.MSELoss() 
# CAE models
encoder = CAE.Encoder()
decoder = CAE.Decoder()
    
# Optimiser
param_to_optimise = [
    {'params': encoder.parameters()},
    {'params': decoder.parameters()}]

optimizer = torch.optim.Adam(param_to_optimise,
                             lr=1e-3, 
                             weight_decay=1e-5)

########################### GPU check ########################################
# Check if the GPU is available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#print(f'Selected device: {device}')

# Move both the encoder and the decoder to the selected device
#model.to(device)


########################### plot images function ########################################
def plot_outputs(encoder, decoder):
    """Plots after final epoch"""
    #obtain one batch of test images
    dataiter = iter(valgen)
    images, _ = next(dataiter)
    
    # get sample outputs
    encoded_data = encoder(images)
    output = decoder(encoded_data) 
    # prep images for display
    images = images.numpy()
    
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
""" 
Uses train_model() and val_model from training.py
loops over num of epochs using training_model() to update weights returns mean loss.
Val_model just returns mean loss
"""
num_epochs = 1000
diz_loss = {'train_loss':[], 'val_loss': []}

for epoch in range(num_epochs):
   train_loss = train_model(encoder, decoder, traingen,device,loss_fn,optimizer)
   val_loss = val_model(encoder, decoder, valgen,device,loss_fn)
   print('\n EPOCH {}/{} \t train loss {} \t val loss {}'.format(epoch + 1, num_epochs,train_loss,val_loss))
   diz_loss['train_loss'].append(train_loss)
   diz_loss['val_loss'].append(val_loss)
plot_outputs(encoder, decoder)
    
#plot the training and validation loss at each epoch
plt.figure(figsize=(10,8))
plt.semilogy(diz_loss['train_loss'], label='Train')
plt.semilogy(diz_loss['val_loss'], label='Valid')
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.legend()
plt.title('Training loss')
plt.show()

######################### Save/ load Model ###################################
"""
# Specify a path
PATH = "entire_model.pt"
# Save
torch.save(model, PATH)
 
# Load
model = torch.load(PATH)
model.eval()
 """   
########################### KDE ########################################

# Encodes training data
encoded_samples = encode_samples(encoder, traingen)

#Fit KDE to the image latent data
kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(encoded_samples)

# retreives one batch to calculate pdf for that batch based on KDE
train_batch = next(iter(traingen)) 
#Get average and std dev. of density and recon 
batch_pdf = calc_density_and_recon_error(encoder, decoder, train_batch, kde, loss_fn)

######################### Outlier detection ###################################

#For this let us generate a batch of images for each. 
sample_batch = next(iter(valgen)) # retreives one batch
# encode that one batch for testing for outliers
encoded_sample_batch = encode_samples(encoder, valgen)

# create empty array tlier classifing
non_outlier=[]
outlier=[]

# Loop through sample batch classifying each image as outlier or not
for im in range(0, sample_batch[1].size(dim=0)-1):
    
    img  = sample_batch[0][im,:,:,:]
    
    result = check_outlier(img, encoder, decoder, kde, batch_pdf, loss_fn)
    if result == 1: 
        outlier.append(img)
    else: non_outlier.append(img)
        
# Plot outlier images
fig, axarr = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, figsize=(4,4))
for idx in range(len(outlier)):
    ax = fig.add_subplot(2, len(outlier)/2, idx+1, xticks=[], yticks=[])
    imshow(outlier[idx])
    ax.set_title("outlier images") 
########################## Plot final scatter ###############################   
# Plot final 
fig, ax = plt.subplots(figsize=(8,8))
ax.scatter(encoded_sample_batch[0], encoded_sample_batch[1])
ax.set_title('sample batch')
plt.show()        
        