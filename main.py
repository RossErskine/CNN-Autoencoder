#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 09:31:36 2022

Main is the interface for which the project CNN Autoencoder will start

Source: https://medium.com/dataseries/convolutional-autoencoder-in-pytorch-on-mnist-dataset-d65145c132ac

@author: Ross Erskine
"""


import matplotlib.pyplot as plt

import torch
from sklearn.neighbors import KernelDensity
import numpy as np
import pandas as pd
from PIL import Image



# import modules
import Parameters as pr
import ImageGenerator as ig
import CNN_Autoencoder as CAE
from training import train_model, val_model
from encode_samples import encode_samples
from density_n_recon_error import calc_density_and_recon_error
#import Kernel_density_estimate as KDE

############################## Constructors ##################################
params = pr.Paramaters(batch_size=512) # Construct Paramaters

test_images = ig.get_test_set()
filename = '../training'
traingen, valgen = ig.train_dataLoader(filename)  # Construct ImageGenerator

loss_fn = torch.nn.MSELoss()

encoder = CAE.Encoder()
decoder = CAE.Decoder()
    
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

##################### helper function ###########################
def imshow(img):
    # helper function to un-normalize and display an image
    img = img / 2 + 0.5  # unnormalize
    plt.imshow(np.transpose(img, (1, 2, 0)))  # convert from Tensor image
    

########################### plot images function ########################################
def plot_outputs(encoder, decoder):
    """Plots after every epoch"""
    #obtain one batch of test images
    dataiter = iter(valgen)
    images, _ = next(dataiter)
    
    # get sample outputs
    encoded_data = encoder(images)
    output = decoder(encoded_data) 
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

num_epochs = 25
diz_loss = {'train_loss':[], 'val_loss': []}

for epoch in range(num_epochs):
   train_loss = train_model(encoder, decoder, traingen,device,loss_fn,optimizer)
   val_loss = val_model(encoder, decoder, valgen,device,loss_fn)
   print('\n EPOCH {}/{} \t train loss {} \t val loss {}'.format(epoch + 1, num_epochs,train_loss,val_loss))
   diz_loss['train_loss'].append(train_loss)
   diz_loss['val_loss'].append(val_loss)
plot_outputs(encoder, decoder)
    
#plot the training and loss at each epoch

plt.figure(figsize=(10,8))
plt.semilogy(diz_loss['train_loss'], label='Train')
plt.semilogy(diz_loss['val_loss'], label='Valid')
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
#plt.grid()
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
train_batch = next(iter(traingen))
encoded_df = encode_samples(encoder, traingen)
encoded_samples = encoded_df.to_numpy()

#encoded_img_vector = [np.reshape(img, (256)) for img in encoded_samples]

#Fit KDE to the image latent data
kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(encoded_samples)

#Get average and std dev. of density and recon. error for uninfected and anomaly (parasited) images. 
#For this let us generate a batch of images for each. 
sample_batch = next(iter(valgen))


batch_pdf = calc_density_and_recon_error(encoder, decoder, train_batch, kde, loss_fn)

######################### Outlier detection ###################################

#Now, input unknown images and sort as Good or Anomaly
def check_anomaly(test_images):
    density_threshold = 2500 #Set this value based on the above exercise
    reconstruction_error_threshold = 0.004 # Set this value based on the above exercise
    img  = Image.open(test_images)
    img = np.array(img.resize((128,128), Image.ANTIALIAS))
    plt.imshow(img)
    img = img / 255.
    img = img[np.newaxis, :,:,:]
    encoded_img = encoder.predict([[img]]) 
    encoded_img = [np.reshape(img, (4,8,8)) for img in encoded_img] 
    density = kde.score_samples(encoded_img)[0] 

    reconstruction = decoder.predict([[img]])
    reconstruction_error = decoder.evaluate([reconstruction],[[img]], batch_size = 1)[0]

    if density < density_threshold or reconstruction_error > reconstruction_error_threshold:
        print("The image is an anomaly")
        
    else:
        print("The image is NOT an anomaly")
        
        
########################## Plot final scatter ###############################   
## TODO with anomolay     
fig, ax = plt.subplots(figsize=(8,8))
ax.scatter(encoded_df[0], encoded_df[1])
plt.show()        
        