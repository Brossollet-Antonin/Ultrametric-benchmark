# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 10:39:18 2019

@author: Antonin
"""


import torch.nn as nn
import torch.optim as optim


def mem_SGD(net, mini_batch, lr, momentum, device):
    inputs, labels = mini_batch
    inputs, labels = inputs.to(device), labels.to(device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    optimizer.zero_grad()
    outputs = net(inputs)

    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    return loss.item()
     