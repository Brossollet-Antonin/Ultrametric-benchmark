# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 11:49:26 2019

@author: Antonin
"""


import torch
import torchvision



def sort_MNIST(train):
    # If True, return sorted train set, else return sorted test set
    train_data_sorted= [[] for i in range(10)]
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./files/', train=train, download=True,
                         transform=torchvision.transforms.Compose([
                           torchvision.transforms.ToTensor(),
                           torchvision.transforms.Normalize(
                             (0.1307,), (0.3081,))
                         ])),
        batch_size=1, shuffle=True)
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        train_data_sorted[int(labels)].append(data)
    return train_data_sorted

def sort_dataset(dataset, train):
    # If True, return sorted train set, else return sorted test set
    if dataset == 'MNIST':
        train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST('./files/', train=train, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
            batch_size=1, shuffle=True)
    elif dataset == 'CIFAR10':
         train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR10('./files/', train=train, download=True,
                             transform=torchvision.transforms.Compose(
                                     [torchvision.transforms.ToTensor(),
                                      torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                      ])),
            batch_size=1, shuffle=True)    
    elif dataset =='CIFAR100':
         train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR100('./files/', train=train, download=True,
                             transform=torchvision.transforms.Compose(
                                     [torchvision.transforms.ToTensor(),
                                      torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                      ])),
            batch_size=1, shuffle=True)  
                               
    train_data_sorted = [[] for i in range(10)] if (dataset=='MNIST' or dataset=='CIFAR10') else [[] for i in range(100)] 
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        train_data_sorted[int(labels.item())].append(data)
    return train_data_sorted    



#
#def sort_MNIST():                
#    train_loader = torch.utils.data.DataLoader(
#            torchvision.datasets.MNIST('/files/', train=True, download=True,
#                             transform=torchvision.transforms.Compose([
#                               torchvision.transforms.ToTensor(),
#                               torchvision.transforms.Normalize(
#                                 (0.1307,), (0.3081,))
#                             ])),
#            batch_size=1, shuffle=True)
#    train_data_sorted= [[] for i in range(10)]
#    for i, data in enumerate(train_loader, 0):
#        inputs, labels = data
#        train_data_sorted[int(labels)].append(data)    
#    return train_data_sorted
            


