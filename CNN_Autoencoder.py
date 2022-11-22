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
        self.encoder_con = nn.Sequential(                       # Layer sizes
            nn.Conv2d(3, 64, 3, stride=2, padding=1),           # 64@ 128x128
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),                                 # 64@ 64x64
            nn.Conv2d(64,32, 3, stride=2, padding=1 ),          # 32@ 64x64
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),                                 # 32@ 32x32
            nn.Conv2d(32, 16, 3, stride=2, padding=1),          # 16@ 32x32
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)                                  # 16@ 16x16
            
        )
        
        
    def forward(self, x):
        x = self.encoder_con(x)
        return x

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
              
        
        # convolutional layers
        self.decoder_con = nn.Sequential(                                  # Layer sizes
            nn.ConvTranspose2d(16, 16, 3, stride=2, padding=1, output_padding=1),     # 16@ 16x16
            nn.ReLU(True),
            nn.MaxUnpool2d(2, 2),                                    # 16@ 32x32
            nn.ConvTranspose2d(16, 32, 3, stride=2, padding=1, output_padding=1),      # 32@ 32x32
            nn.ReLU(True),
            nn.MaxUnpool2d(2, 2),                                    # 32@ 64x64
            nn.ConvTranspose2d(32, 64, 3, stride=2, padding=1 , output_padding=1),     # 64@ 64x64
            nn.ReLU(True),
            nn.MaxUnpool2d(2, 2),                                    # 64@ 128x128
            nn.ConvTranspose2d(64, 3, 3, stride=2, padding=0 ),      # 3@ 128x128
                        
            
        )
        
    def forward(self, x):
        x = self.decoder_con(x)
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