#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 16:45:26 2019

@author: diaa
"""

from matplotlib import pyplot as plt

def display_graph(train_losses, valid_losses, train_accs, valid_accs):
    plt.plot(train_losses, label='Training loss')
    plt.plot(valid_losses, label='Validation loss')
    plt.plot(train_accs, label='Train accuracy')
    plt.plot(valid_accs, label='Validation accuracy')
    plt.legend(frameon=False)