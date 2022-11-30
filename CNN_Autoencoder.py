#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CNN Autoencoder Class


Created on Mon Nov 14 09:39:52 2022
Tested: TODO
Last commited: 

Source: https://github.com/patrickloeber/pytorch-examples/blob/master/Autoencoder.ipynb
@author: Ross Erskine ppxre1
"""

import torch
from torch import nn
SIZE = 128

          

class Convolutional_Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Encoder layers
        # N, 3, 128, 128
        self.encoder = nn.Sequential(                       
            nn.Conv2d(3, 32, 3, stride=2, padding=1),    # -> N, 32, 64, 64
            nn.ReLU(True),
                                             
            nn.Conv2d(32 ,16, 3, stride=2, padding=1 ),  # -> N, 16, 32, 32
            nn.ReLU(True),
            
            nn.Conv2d(16 ,8, 3, stride=2, padding=1 ),  # -> N, 8, 16, 16
            nn.ReLU(True),
       
            nn.Conv2d(8, 4, 3, stride=2, padding=1)   # -> N, 4, 8, 8
        )
        
        # Encoder layers
        # N, 4, 8, 8
        self.decoder = nn.Sequential(  
            nn.ConvTranspose2d(4, 8, 3, stride=2, padding=1, output_padding=1), # -> N, 8, 16, 16
            nn.ReLU(True),
            
            nn.ConvTranspose2d(8, 16, 3, stride=2, padding=1, output_padding=1), # -> N, 16, 32, 32
            nn.ReLU(True),
            
            nn.ConvTranspose2d(16, 32, 3, stride=2, padding=1, output_padding=1), # -> N, 32, 64, 64
            nn.ReLU(True),
            
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1), # -> N, 3, 128, 128
            nn.Sigmoid()
        )
        
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def get_encoded(self, x):
        encoded = self.encoder(x)
        return encoded



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
            CAE = Convolutional_Autoencoder()
            print(CAE)
            
        def test_get_ecoder(self):
            """Test Get encoder function """
            model = Convolutional_Autoencoder()
            encoder = model.get_encoder()
            print(encoder)
            
            
            
    unittest.main()   