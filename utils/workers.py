#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 16:45:13 2019

@author: diaa
"""
from time import time

import numpy as np

import torch



def train(model, train_loader, valid_loader,n_epochs, optimizer, loss_function, model_file):
    
    train_losses = []
    train_accs = []

    valid_losses = []
    valid_accs = []

    min_valid_loss = np.Inf
    model.train()      
    for e in range(1,n_epochs+1):
        epoch_start = time()
        batch_number = 0

        train_loss = 0
        train_acc = 0
        batch_start = time()
        for x,y in train_loader:
            batch_number += 1
           
            if torch.cuda.is_available() :
                x, y = x.cuda(), y.cuda()

            optimizer.zero_grad()

            y_ = model.forward(x)

            loss, y_,y= loss_function(y_.cpu(),y.cpu())

            loss.backward()
            
            optimizer.step()
            
            train_loss += loss.item()

            ps = torch.exp(y_)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == y.view(*top_class.shape)
            train_acc += torch.mean(equals.type(torch.FloatTensor))
            delay = time()-batch_start
           
            print("batch size : {}\nbatch per total no of batches : {}/{} \
            \ntrain batch finished : {:.3f} % \ntime left : {}s \ndelay : {}s \
            \nloss : {}\n\n".format(len(x), batch_number, len(train_loader),
            batch_number/len(train_loader) *100., delay * (len(train_loader)-\
            batch_number), delay, loss.item()))
            
            batch_start = time()
            

        
        valid_loss = 0
        valid_acc = 0
        with torch.no_grad():
            model.eval()

            batch_number = 0
            batch_start = time()
            for x,y in valid_loader:
                batch_number += 1
                if torch.cuda.is_available() :
                    x, y = x.cuda(), y.cuda()
                x = x.squeeze()
                y = y.squeeze()
                y_ = model.forward(x)

                loss, y_,y= loss_function(y_.cpu(),y.cpu())

                valid_loss += loss.item()

                ps = torch.exp(y_)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == y.view(*top_class.shape)

                valid_acc += torch.mean(equals.type(torch.FloatTensor))
                delay = time()-batch_start
                
                print("batch size : {}\nbatch per total no of batches : {}/{} \
                \ntrain batch finished : {:.3f} % \ntime lift : {} s\
                \ndelay : {}s \nloss : {}\n\n".format(len(x),
                batch_number, len(valid_loader), batch_number/len(valid_loader)\
                      *100., delay * (len(valid_loader)-batch_number), delay,\
                                                      loss.item()))
           
                batch_start = time()
                

        train_loss /= len(train_loader)     
        train_acc /= len(train_loader)  

        valid_loss /= len(valid_loader)
        valid_acc /= len(valid_loader)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        valid_losses.append(valid_loss)
        valid_accs.append(valid_acc)

        if min_valid_loss > valid_loss:
            print ('Validation loss decreased ({:.6f} --> {:.6f}). \
            Saving model ...\n'.format(min_valid_loss, valid_loss))
            min_valid_loss = valid_loss
            torch.save(model.state_dict(), model_file)
        
        delay = time() - epoch_start


        print("Epoch : {} \nTrain Finished : {:.3f} %\nTime Left : {:.3f} s\
        \nTraining Loss : {:.6f} \nValidation Loss : {:.6f} \nTrain Accuracy :\
        {:.3f} %\nValidation Accuracy : {:.6f} %\nDelay : {:.6f} s \n\n".format(
            e,e / n_epochs * 100., delay * (n_epochs - e) ,train_loss, valid_loss
            ,train_acc,valid_acc,delay))
        
    return train_losses, train_accs, valid_losses, valid_accs




def test(model, test_loader, loss_function):
   
    test_loss = 0
    test_acc = 0
    
    model.eval()

    with torch.no_grad():

        batch_number = 0
        batch_start = time()
        for x,y in test_loader:
            batch_number += 1
            if torch.cuda.is_available() :
                x, y = x.cuda(), y.cuda()
            x = x.squeeze()
            y = y.squeeze()
            y_ = model.forward(x)

            loss, y_,y= loss_function(y_.cpu(),y.cpu())

            test_loss += loss.item()

            ps = torch.exp(y_)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == y.view(*top_class.shape)

            test_acc += torch.mean(equals.type(torch.FloatTensor))
            delay = time()-batch_start

            print("batch size : {}\nbatch per total no of batches : {}/{} \
            \ntrain batch finished : {:.3f} % \ntime lift : {} s\
            \ndelay : {}s \nloss : {}\n\n".format(len(x),
            batch_number, len(test_loader), batch_number/len(test_loader)\
                  *100., delay * (len(test_loader)-batch_number), delay,\
                                                  loss.item()))
            batch_start = time()
            
    return test_loss, test_acc