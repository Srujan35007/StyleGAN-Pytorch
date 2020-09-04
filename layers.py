'''
Author  :   Suddala Srujan
Date    :   4th Sep 2020

Contains custom layers for Generator and Discriminator
'''

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

class Upsample(nn.Module):
    def __init__(self, res):
        super(Upsample, self).__init__()
        self.res_threshold = 128
        if res < self.res_threshold:
            self.up = nn.Upsample(scale_factor=2, mode='nearest')
        else:
            # TODO : get relationship b/w channels and current resolution
            self.up = nn.ConvTranspose2d(TODO, TODO, 4,2,1)
    
    def __call__(self, x):
        self.up(x)