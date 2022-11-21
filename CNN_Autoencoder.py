#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CNN Autoencoder Class


Created on Mon Nov 14 09:39:52 2022
Tested: TODO
Last commited: 

Source: https://medium.com/dataseries/convolutional-autoencoder-in-pytorch-on-mnist-dataset-d65145c132ac
@author: Ross Erskine ppxre1
"""

import torch
from torch import nn
SIZE = 128

          

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        # convolutional layers
        self.encoder_con = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 32, 3, stride=2, padding=1 ),
            nn.ReLU(True),
            nn.Conv2d(32, 16, 3, stride=2, padding=1 ),
            nn.ReLU(True)
        )
        ## Flatten layers
        self.flatten = nn.Flatten(start_dim=1)
        
        ## Linear layers
        self.encoder_lin = nn.Sequential(
            nn.Linear(3 * 3 * 16, 128),
            nn.ReLU(True),
            nn.Linear(128, 4)
        )
        
    def forward(self, x):
        x = self.encoder_con(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
              
               
        ## Linear layers
        self.Decoder_lin = nn.Sequential(
            nn.Linear(128,3*3*16),
            nn.ReLU(True),
            nn.Linear(4, 128)
        )
        # Flatten layers
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(16, 3, 3))
        # convolutional layers
        self.Decoder_con = nn.Sequential(
            nn.ConvTranspose2d(16, 32, 3, stride=2, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 64, 3, stride=2, output_padding=1 ),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 3, stride=2, output_padding=1 ),
            nn.ReLU(True)
        )
        
    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x

if __name__ == '__main__': 
    
############################# Testing #####################################
    
    """
    Test class for CNN_Autoencoder
    """
    import unittest     
    
    class Test_CNN_Autoencoder(unittest.TestCase):
        """Test CNN Autoencoder"""
        
        def test_constructor(self):
            """Test CNN Autoencoder constructor"""
            #TODO
            
            
            
    unittest.main()   