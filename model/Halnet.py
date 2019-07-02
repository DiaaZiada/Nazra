#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sun May  5 21:30:57 2019

@author: diaa
"""

import torch 
from torch import nn

class ConvBatchReluBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0):
        super(ConvBatchReluBlock, self).__init__()
        
        self.conv =  nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=kernel_size, stride=stride,
                      padding=padding)
        self. batchnorm = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()
        
        if torch.cuda.is_available():
            self.cuda()
        
    def forward(self, inp):
        
        x = self.conv(inp)
        x = self.batchnorm(x)
        x = self.relu(x)
        
        return x
    
class HALNetResConvSequence(nn.Module):
    
    def __init__(self, stride, filters1, filters2, padding1=1, padding2=0, 
                 padding3=0, first_in_channels=0):
        super(HALNetResConvSequence, self).__init__()

        if first_in_channels == 0:
             first_in_channels = filters1
        
        self.conv1 = ConvBatchReluBlock(kernel_size=1, stride=stride, out_channels=filters1,
                              in_channels=first_in_channels, padding=padding1)
        self.conv2 = ConvBatchReluBlock(kernel_size=3, stride=1, out_channels=filters1,
                              in_channels=filters1, padding=padding2)
        self.conv3 = ConvBatchReluBlock(kernel_size=1, stride=1, out_channels=filters2,
                              in_channels=filters1, padding=padding3)
        
        if torch.cuda.is_available():
            self.cuda()
            
    def forward(self, inp):
        
        x = self.conv1(inp)
        x = self.conv2(x)
        x = self.conv3(x)
        
        return x
    
class HALNetResBlockIDSkip(nn.Module):
  
    def __init__(self, filters1, filters2,
                 padding_right1=1, padding_right2=0, padding_right3=0):
        super(HALNetResBlockIDSkip, self).__init__()
        self.right_res = HALNetResConvSequence(stride=1,
                                               filters1=filters1,
                                               filters2=filters2,
                                               padding1=padding_right1,
                                               padding2=padding_right2,
                                               padding3=padding_right3,
                                               first_in_channels=
                                               filters2)
        self.relu = nn.ReLU()

        if torch.cuda.is_available():
            self.cuda()
            
    def forward(self, inp):
        left_res = inp
        right_res = self.right_res(inp)
        # element-wise sum
        out = left_res + right_res
        out = self.relu(out)
        return out

class HALNetResBlockConv(nn.Module):
    def __init__(self, stride, filters1, filters2, first_in_channels=0,
                 padding_left=0, padding_right1=0, padding_right2=0,
                 padding_right3=0):
        super(HALNetResBlockConv, self).__init__()
        
        if first_in_channels == 0:
            first_in_channels = filters1
        self.left_res = ConvBatchReluBlock(kernel_size=1, stride=stride,
                                              out_channels=filters2,
                                              padding=padding_left,
                                              in_channels=first_in_channels)
        
        self.right_res = HALNetResConvSequence(stride=stride,
                                               filters1=filters1,
                                               filters2=filters2,
                                               padding1=padding_right1,
                                               padding2=padding_right2,

                                               padding3=padding_right3,
                                               first_in_channels=
                                               first_in_channels)
        self.relu = nn.ReLU()

        if torch.cuda.is_available():
            self.cuda()
            
    def forward(self, inp):
        left_res = self.left_res(inp)
        right_res = self.right_res(inp)
        # element-wise sum
        out = left_res + right_res
        out = self.relu(out)
        return out
    
    


class HALNet(nn.Module):
    
    def __init__(self):
        super(HALNet, self).__init__()
      
        
        self.conv1 = ConvBatchReluBlock(kernel_size=7, stride=1, out_channels=64,
                                                  in_channels=4, padding=3)
        self.mp1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.res2a = HALNetResBlockConv(stride=1, filters1=64, filters2=256,
                                        padding_right1=1)
        self.res2b = HALNetResBlockIDSkip(filters1=64, filters2=256)
        self.res2c = HALNetResBlockIDSkip(filters1=64, filters2=256)
        self.res3a = HALNetResBlockConv(stride=2, filters1=128, filters2=512,
                                               padding_right3=1, first_in_channels=256)
      
        self.res3b = HALNetResBlockIDSkip(filters1=128, filters2=512)
        self.res3c = HALNetResBlockIDSkip(filters1=128, filters2=512)
   
        self.res4a = HALNetResBlockConv(stride=2, filters1=256, filters2=1024,
                                        padding_right3=1,
                                        first_in_channels=512)            
        self.res4b = HALNetResBlockIDSkip(filters1=256, filters2=1024)
        self.res4c = HALNetResBlockIDSkip(filters1=256, filters2=1024)
        self.res4d = HALNetResBlockIDSkip(filters1=256, filters2=1024)
        self.conv4e = ConvBatchReluBlock(kernel_size=3, stride=1, out_channels=512,
                                                   in_channels=1024, padding=1)
      
        self.conv4f = ConvBatchReluBlock(kernel_size=3, stride=1, out_channels=256,
                                                   in_channels=512, padding=1)
        if torch.cuda.is_available():
            self.cuda()
            
    def forward(self, x):
        
        out = self.conv1(x)
        out = self.mp1(out)
        out = self.res2a(out)
        out = self.res2b(out)
        out = self.res2c(out)
        res3aout = self.res3a(out)
        out = self.res3b(res3aout)
        out = self.res3c(out)
        res4aout = self.res4a(out)
        out = self.res4b(res4aout)
        out = self.res4c(out)
        out = self.res4d(out)
        conv4eout = self.conv4e(out)
        conv4fout = self.conv4f(conv4eout)
        return res3aout, res4aout, conv4eout, conv4fout

        
halnet = HALNet()
print(halnet)   
        
    
