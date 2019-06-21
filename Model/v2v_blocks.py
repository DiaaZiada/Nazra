# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 01:09:03 2019

@author: Diaa Elsayed
"""

import torch
from torch import nn

class BasicBlock(nn.Module):
    
    def __init__(self,in_channels,out_channels,kernel_size,strid=1):
        super(BasicBlock, self).__init__()
        
        padding = (kernel_size - 1) // 2
        self.conv3d = nn.Conv3d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU()
       
        if torch.cuda.is_available():
            self.cuda()
            
    def forward(self,x):
        
        x = self.conv3d(x)
        x = self.bn(x)
        x = self.relu(x)
        
        return x
    
class ResBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        
        self.conv3d_1 = nn.Conv3d(in_channels=in_channels, 
                                  out_channels=out_channels,
                                  kernel_size=3, stride=1,
                                  padding = 1)
        self.bn_1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU()
        
        self.conv3d_2 = nn.Conv3d(in_channels=out_channels, 
                                  out_channels=out_channels,
                                  kernel_size=3, stride=1,
                                  padding = 1) 
        self.bn_2 = nn.BatchNorm3d(out_channels)
        
        if in_channels == out_channels:
            self.skip = False
        else:
            self.skip = True
            self.conv3d_skip = nn.Conv3d(in_channels=in_channels, 
                                  out_channels=out_channels,
                                  kernel_size=1, stride=1,
                                  padding = 0)
            self.bn_skip = nn.BatchNorm3d(out_channels)
    
        if torch.cuda.is_available():
            self.cuda()
            
    def forward(self, x) :
        
        res = self.conv3d_1(x)
        res = self.bn_1(res)
        res = self.relu(res)
        res = self.conv3d_2(res)
        res = self.bn_2(res)
        
        if self.skip:
            skip = self.conv3d_skip(x)
            skip = self.bn_skip(skip)
            res += skip
            
        res = self.relu(res)
        
        return res
    
class DonwnsampleBlock(nn.Module):
    
    def __init__(self, kernel_size):
        super(DonwnsampleBlock, self).__init__()
        
        self.pool = nn.MaxPool3d(kernel_size=kernel_size)
        
        if torch.cuda.is_available():
            self.cuda()
            
    def forward(self, x):
        x = self.pool(x)
        return x
            
class UpsampleBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2):
        super(UpsampleBlock, self).__init__()
        
        self.conv_T3d = nn.ConvTranspose3d(in_channels=in_channels,
                                           out_channels=out_channels,
                                           kernel_size=kernel_size, 
                                           stride=stride,
                                           padding=0,
                                           output_padding=0)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU()
        
        if torch.cuda.is_available():
            self.cuda()
            
    def forward(self, x):
        
        x = self.conv_T3d(x)
        x = self.bn(x)
        x = self.relu(x)
        
        return x        
    
    