# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 11:14:53 2019

@author: Antonin
"""

import torch.nn as nn
import torch.nn.functional as F



# 2 hidden layers MLP with 256 ReLU units in each layers (similar to Chaudhry et al. (2019))
input_size = 100
output_size = 128
hidden_size = 250

class Net_FCRU(nn.Module):
    def __init__(self):
        super(Net_FCRU, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size,hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = x.view(-1, input_size)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def layer2(self, x):
        x = x.view(-1, input_size)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def layer1(self, x):
        x = x.view(-1, input_size)
        x = self.fc1(x)
        return x
    
    def layer3(self, x):
        x = x.view(-1, input_size)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x   
    
    def layer4(self, x):
        x = x.view(-1, input_size)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    
    
# 2 hidden layers CNN

class Net_CNN(nn.Module):
    def __init__(self, dataset):
        super(Net_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1,6,5) if dataset=='MNIST' else nn.Conv2d(3,6,5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(16*4*4, 120) if dataset=='MNIST' else nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84,100) if dataset=='CIFAR100' else nn.Linear(84,10)
        self.dataset = dataset
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*4*4) if self.dataset=='MNIST' else x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
hidden_size = 100

class Net_FCL(nn.Module):
    def __init__(self):
        super(Net_FCL, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size,output_size)
        
    def forward(self, x):
        x = x.view(-1, input_size)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
    
    

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


net = Net_CNN('MNIST')        
nbr_para = count_parameters(net)
print(nbr_para)