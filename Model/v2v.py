# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 01:08:50 2019

@author: Diaa Elsayed
"""
import torch
from torch import nn
from v2v_blocks import *

class FrontLayers(nn.Module):
    
    def __init__(self,in_channels):
        super(FrontLayers, self).__init__()
        
        self.basic = BasicBlock(in_channels=in_channels,out_channels=16, kernel_size=7)
        
        self.pool = DonwnsampleBlock(kernel_size=2)
        
        self.res1 = ResBlock(in_channels=16, out_channels=32)
        self.res2 = ResBlock(in_channels=32, out_channels=32)
        self.res3 = ResBlock(in_channels=32, out_channels=32)
        
        if torch.cuda.is_available():
            self.cuda()
            
    def forward(self, x):
        
        x = self.basic(x)
        x = self.pool(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
       
        return x
    
class EncoderDecorder(nn.Module):

    def __init__(self):
        super(EncoderDecorder, self).__init__()
        
        self.encoder_pool1 = DonwnsampleBlock(kernel_size=2)
        self.encoder_res1 = ResBlock(in_channels=32, out_channels=64)
        self.encoder_pool2 = DonwnsampleBlock(kernel_size=2)
        self.encoder_res2 = ResBlock(in_channels=64, out_channels=128)
        
        self.mid_res = ResBlock(in_channels=128, out_channels=128)

        self.decoder_res2 = ResBlock(in_channels=128, out_channels=128)
        self.decoder_upsample2 = UpsampleBlock(in_channels=128, out_channels=64,
                                               kernel_size=2, stride=2)
        self.decoder_res1 = ResBlock(in_channels=64, out_channels=64)
        self.decoder_upsample1 = UpsampleBlock(in_channels=64, out_channels=32,
                                               kernel_size=2, stride=2)

        self.skip_res1 = ResBlock(in_channels=32, out_channels=32)
        self.skip_res2 = ResBlock(in_channels=64, out_channels=64)
        
        if torch.cuda.is_available():
            self.cuda()
            
    def forward(self, x):
        
        skip_x1 = self.skip_res1(x)
        x = self.encoder_pool1(x)
        x = self.encoder_res1(x)
        skip_x2 = self.skip_res2(x)
        x = self.encoder_pool2(x)
        x = self.encoder_res2(x)

        x = self.mid_res(x)

        x = self.decoder_res2(x)
        x = self.decoder_upsample2(x)
        x = x + skip_x2
        x = self.decoder_res1(x)
        x = self.decoder_upsample1(x)
        x = x + skip_x1

        return x
    
class BackLayers(nn.Module):
    
    def __init__(self, out_channels):
        super(BackLayers, self).__init__()

        self.res =  ResBlock(in_channels=32, out_channels=32)
        self.basic1 = BasicBlock(in_channels=32,out_channels=32, kernel_size=1)
        self.basic2 = BasicBlock(in_channels=32,out_channels=32, kernel_size=1)
        

#         self.output_layer = nn.Conv3d(32, out_channels, kernel_size=1, stride=1, padding=0)
        self.output_layer = UpsampleBlock(32, out_channels, kernel_size=2, stride=2,)

        if torch.cuda.is_available():
            self.cuda()
            
    def forward(self, x):
       
        x = self.res(x)
        x = self.basic1(x)
        x = self.basic2(x)
        x = self.output_layer(x)
        
        return(x)
        
class V2VModel(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super(V2VModel, self).__init__()
        
        self.front_layers = FrontLayers(in_channels=in_channels)
        self.encoder_decoder = EncoderDecorder()
        self.back_layers = BackLayers(out_channels=out_channels)
    
        if torch.cuda.is_available():
            self.cuda()
            
    def forward(self, x):
        x = self.front_layers(x)
        x = self.encoder_decoder(x)
        x = self.back_layers(x)
        return x