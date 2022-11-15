#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 09:31:36 2022

Main is the interface for which the project CNN Autoencoder will start

@author: Ross Erskine
"""


import matplotlib.pyplot as plt


# import modules
import Parameters as pr
import ImageGenerator as ig
import CNN_Autoencoder as CAE

params = pr.Paramaters()
datagen = ig.train_imageGenerator()

model = CAE.CAE_model()
model.summary()


history = model.fit(datagen, steps_per_epoch= 32 // params.get_batch_size(),epochs=32)

#plot the training and validation accuracy and loss at each epoch
loss = history.history['loss']
#val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
#plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
