# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 11:13:48 2019

@author: Antonin
"""

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import rates_correlation
import numpy as np
from random import randint

 
class Net_preproc(nn.Module):
    def __init__(self):
        super(Net_preproc, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256,256)
        self.fc3 = nn.Linear(256,10)
        
    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def layer_extract(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
#        x = F.relu(self.fc2(x))
        x= self.fc2(x)
        return x

def train_preproc():
    train_epoch = 5
    learning_rate = 0.01
    momentum = 0.5 
        
        
    train_loader = torch.utils.data.DataLoader(
                torchvision.datasets.MNIST('/files/', train=True, download=True,
                                 transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                     (0.1307,), (0.3081,))
                                 ])),
                batch_size=64, shuffle=True)
                                   
    test_loader = torch.utils.data.DataLoader(
                torchvision.datasets.MNIST('/files/', train=False, download=True,
                                 transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                     (0.1307,), (0.3081,))
                                 ])),
                batch_size=1, shuffle=True)
                                   
    
    preprocessor = Net_preproc()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(preprocessor.parameters(), lr=learning_rate, momentum=momentum)
    
    print("--- Training preprocessor ---")
    for epoch in range(train_epoch):
        running_loss = 0.0
        for i,data in enumerate(train_loader):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = preprocessor(inputs)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 99:    # print every 100 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0
                
    print('--- Data preprocessor end training, validation starts --')
    
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = preprocessor(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    
    print('Accuracy of the preprocessor on the %d test images: %d %%' % (total,
        100 * correct / total))
    return preprocessor


def preproc_corr(net, data, nbr_avg):
    length = len(data)
    cor_matrix = np.zeros((length, length))   
    for i in range(length):
        for j in range(length):
            temp_cor = 0 
            for k in range(nbr_avg):
                indA, indB = [randint(0,len(data[i])-1) for a in range(10)], [randint(0,len(data[j])-1) for a in range(10)]
                for n in indA:
                    for m in indB:
                        temp_cor += rates_correlation.im_cor(
                                net.layer_extract(data[i][n][0]).detach(), 
                                net.layer_extract(data[j][m][0]).detach())
            cor_matrix[i][j] = temp_cor/(10*10*nbr_avg)
    return cor_matrix

def rates_preproc(net, data, T, nbr_avg):
    cor_matrix = preproc_corr(net, data, nbr_avg)
    rates_matrix = np.zeros(cor_matrix.shape)
    for i in range(cor_matrix.shape[0]):
        for j in range(cor_matrix.shape[1]):
            if i != j: 
                rates_matrix[i][j] = np.exp(-(1-cor_matrix[i][j])/T)
    rates_matrix = (rates_matrix + np.transpose(rates_matrix))/2
    for i in range(cor_matrix.shape[0]):
        sumrow = np.sum(rates_matrix[i])
        rates_matrix[i][i] = 1-sumrow   
        if rates_matrix[i][i] < 0:
            raise ValueError("Temperature too high, selfrates inferior to 0. Lower temperature")
    return rates_matrix

    
