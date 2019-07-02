# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 00:22:22 2019

@author: Diaa Elsayed
"""

def cudafy(object, use_cuda):
    if use_cuda:
        return object.cuda()
    else:
        return object

def dims_fun(w, h):
  
    w, h = (w-7+2*3)/1+1, (h-7+2*3)/1+1 #conv1
    w, h = (w-3+2*1)/2+1, (h-3+2*1)/2+1 #maxpool
    w, h = (w-1+2*0)/1+1, (h-1+2*0)/1+1 #res2out
    w, h = (w-1+2*0)/2+1, (h-1+2*0)/2+1 #res3out
    w,h = int(w),int(h)
    res3out = w * h * 512
   
    w, h = (w-1+2*0)/2+1, (h-1+2*0)/2+1 #res4out
    w,h = int(w),int(h)
    res4out = w * h * 1024
   
    w, h = (w-3+2*1)/1+1, (h-3+2*1)/1+1 #conv4e
    w,h = int(w),int(h)
    conv4e = w*h*512

    w, h = (w-3+2*1)/1+1, (h-3+2*1)/1+1 #conv4f
    w,h = int(w),int(h)
    conv4f = w*h*256
    return res3out,res4out,conv4e,conv4f,h,w
