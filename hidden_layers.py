# -*- coding: utf-8 -*-
"""
Created on Mon May 13 16:13:40 2019

@author: Antonin
"""

from random import shuffle
import torch
import numpy as np
#import matplotlib.pyplot as plt

output_sz = 128

def hidden_rep2_GT(net, dataset, device):
    hidden = [[] for i in range(dataset.branching**dataset.depth)]
    
    j = 0
    test_sequence=[]
    iterator = [iter(dataset.test_data[k]) for k in range(len(dataset.test_data))]
    for i in range(len(dataset.test_data)):
        for j in range(dataset.class_sz_test):
            test_sequence.append(next(iterator[i]))
    shuffle(test_sequence)
    
    with torch.no_grad():
        for data in test_sequence:
            samples, labels = data
            samples, labels = samples.to(device), labels.to(device)
            layer2_outputs = net.layer2(samples)
            hidden[labels].append(layer2_outputs)
    return hidden
    
#    for i in range(len(hidden)):
#        hidde
        


#def hidden_rep2_pred(net, dataset, device):
    
#def avg_hidden_rep(hidden):
#    concat = [hidden[i][0] for i in range(len(hidden))]
#    for i in range(len(hidden)):
#        for tens in hidden[i][1:]:
#            concat[i] = torch.cat((concat[i], tens), dim=0)
#        concat[i] = torch.mean(concat[i], 0)
#        concat[i] = concat[i].numpy()
#    return concat
        
def avg_hidden_rep(hidden):
    concat = []
    for i in range(len(hidden)):
        if len(hidden[i]) != 0:
            concat.append(hidden[i][0])
        else:
            concat.append(torch.zeros(1, output_sz, dtype=torch.float))    
    for i in range(len(hidden)):
        for tens in hidden[i][1:]:
            concat[i] = torch.cat((concat[i], tens), dim=0)
        concat[i] = torch.mean(concat[i], 0) 
        concat[i] = torch.unsqueeze(concat[i],1).numpy()
    return concat            
    
    
#temp2=hidden_rep2_GT(netfc_shuffle, dataset, device)
#average2=avg_hidden_rep(temp2)
#
#average2=np.array(average2)
#plt.figure()
#plt.imshow(average2)
#plt.title('Layer 2')



def hidden_rep1_GT(net, dataset, device):
    hidden = [[] for i in range(dataset.branching**dataset.depth)]
    
    j = 0
    test_sequence=[]
    iterator = [iter(dataset.test_data[k]) for k in range(len(dataset.test_data))]
    for i in range(len(dataset.test_data)):
        for j in range(dataset.class_sz_test):
            test_sequence.append(next(iterator[i]))
    shuffle(test_sequence)
    
    with torch.no_grad():
        for data in test_sequence:
            samples, labels = data
            samples, labels = samples.to(device), labels.to(device)
            layer1_outputs = net.layer1(samples)
            hidden[labels].append(layer1_outputs)
    return hidden
    
#    for i in range(len(hidden)):
#        hidde
        


#def hidden_rep2_pred(net, dataset, device):
    
        
            
    
    
#temp1=hidden_rep1_GT(netfc_shuffle, dataset, device)
#average1=avg_hidden_rep(temp1)
#
#average1=np.array(average1)
#plt.figure()
#plt.imshow(average1)
#plt.title("Layer 1")



def hidden_rep3_GT(net, dataset, device):
    hidden = [[] for i in range(dataset.branching**dataset.depth)]
    
    j = 0
    test_sequence=[]
    iterator = [iter(dataset.test_data[k]) for k in range(len(dataset.test_data))]
    for i in range(len(dataset.test_data)):
        for j in range(dataset.class_sz_test):
            test_sequence.append(next(iterator[i]))
    shuffle(test_sequence)
    
    with torch.no_grad():
        for data in test_sequence:
            samples, labels = data
            samples, labels = samples.to(device), labels.to(device)
            layer3_outputs = net.layer3(samples)
            hidden[labels].append(layer3_outputs)
    return hidden


def hidden_rep4_GT(net, dataset, device):
    hidden = [[] for i in range(dataset.branching**dataset.depth)]
    
    j = 0
    test_sequence=[]
    iterator = [iter(dataset.test_data[k]) for k in range(len(dataset.test_data))]
    for i in range(len(dataset.test_data)):
        for j in range(dataset.class_sz_test):
            test_sequence.append(next(iterator[i]))
    shuffle(test_sequence)
    
    with torch.no_grad():
        for data in test_sequence:
            samples, labels = data
            samples, labels = samples.to(device), labels.to(device)
            layer4_outputs = net.layer4(samples)
            hidden[labels].append(layer4_outputs)
    return hidden


def plot_hidden_GT(net, dataset, device, name, plot):
    temp1 = hidden_rep1_GT(net, dataset, device)
    average1 = avg_hidden_rep(temp1)
    average1 = np.array(average1)
           
    temp2 = hidden_rep2_GT(net, dataset, device)
    average2 = avg_hidden_rep(temp2)   
    average2 = np.array(average2)

    temp3=hidden_rep3_GT(net, dataset, device)
    average3=avg_hidden_rep(temp3)
    average3=np.array(average3)
    
    temp4=hidden_rep4_GT(net, dataset, device)
    average4=avg_hidden_rep(temp4)
    average4=np.array(average4)

    compt = [len(temp1[i]) for i in range(len(temp1))]

    if plot:
     plt.figure()
     plt.imshow(average1)
     plt.title("%s Layer 1 Ground Truth" % name)
     plt.figure()
     plt.imshow(average2)
     plt.title("%s Layer 2 Ground Truth" % name)
     plt.figure()
     plt.imshow(average3)
     plt.title("%s Layer 3 Ground Truth" % name)
     plt.figure()
     plt.imshow(average4)
     plt.title("%s Layer 4 Ground Truth" % name)
    
    return (average1, average2, average3, average4, compt)



""" Average on predicted labels """

def hidden_rep1_pred(net, dataset, device):
    hidden = [[] for i in range(dataset.branching**dataset.depth)]
    
    j = 0
    test_sequence=[]
    iterator = [iter(dataset.test_data[k]) for k in range(len(dataset.test_data))]
    for i in range(len(dataset.test_data)):
        for j in range(dataset.class_sz_test):
            test_sequence.append(next(iterator[i]))
    shuffle(test_sequence)
    
    with torch.no_grad():
        for data in test_sequence:
            samples, labels = data
            samples, labels = samples.to(device), labels.to(device)
            layer1_outputs = net.layer1(samples)
            outputs = net(samples)
            _, pred = torch.max(outputs.data, 1)
            hidden[pred].append(layer1_outputs)
    return hidden

def hidden_rep2_pred(net, dataset, device):
    hidden = [[] for i in range(dataset.branching**dataset.depth)]
    
    j = 0
    test_sequence=[]
    iterator = [iter(dataset.test_data[k]) for k in range(len(dataset.test_data))]
    for i in range(len(dataset.test_data)):
        for j in range(dataset.class_sz_test):
            test_sequence.append(next(iterator[i]))
    shuffle(test_sequence)
    
    with torch.no_grad():
        for data in test_sequence:
            samples, labels = data
            samples, labels = samples.to(device), labels.to(device)
            layer2_outputs = net.layer2(samples)
            outputs = net(samples)
            _, pred = torch.max(outputs.data, 1)
            hidden[pred].append(layer2_outputs)
    return hidden

def hidden_rep3_pred(net, dataset, device):
    hidden = [[] for i in range(dataset.branching**dataset.depth)]
    
    j = 0
    test_sequence=[]
    iterator = [iter(dataset.test_data[k]) for k in range(len(dataset.test_data))]
    for i in range(len(dataset.test_data)):
        for j in range(dataset.class_sz_test):
            test_sequence.append(next(iterator[i]))
    shuffle(test_sequence)
    
    with torch.no_grad():
        for data in test_sequence:
            samples, labels = data
            samples, labels = samples.to(device), labels.to(device)
            layer3_outputs = net.layer3(samples)
            outputs = net(samples)
            _, pred = torch.max(outputs.data, 1)
            hidden[pred].append(layer3_outputs)
    return hidden

def hidden_rep4_pred(net, dataset, device):
    hidden = [[] for i in range(dataset.branching**dataset.depth)]
    
    j = 0
    test_sequence=[]
    iterator = [iter(dataset.test_data[k]) for k in range(len(dataset.test_data))]
    for i in range(len(dataset.test_data)):
        for j in range(dataset.class_sz_test):
            test_sequence.append(next(iterator[i]))
    shuffle(test_sequence)
    
    with torch.no_grad():
        for data in test_sequence:
            samples, labels = data
            samples, labels = samples.to(device), labels.to(device)
            layer4_outputs = net.layer4(samples)
            outputs = net(samples)
            _, pred = torch.max(outputs.data, 1)
            hidden[pred].append(layer4_outputs)
    return hidden


def plot_hidden_pred(net, dataset, device, name, plot):
    temp1 = hidden_rep1_pred(net, dataset, device)
    average1 = avg_hidden_rep(temp1)
    average1 = np.array(average1)
           
    temp2 = hidden_rep2_pred(net, dataset, device)
    average2 = avg_hidden_rep(temp2)   
    average2 = np.array(average2)

    temp3=hidden_rep3_pred(net, dataset, device)
    average3=avg_hidden_rep(temp3)
    average3=np.array(average3)

    temp4=hidden_rep4_pred(net, dataset, device)
    average4=avg_hidden_rep(temp4)
    average4=np.array(average4)
    
    compt = [len(temp1[i]) for i in range(len(temp1))]
    
    if plot:
         plt.figure()
         plt.imshow(average1)
         plt.title("%s Layer 1 Predicted" % name)
         plt.figure()
         plt.imshow(average2)
         plt.title("%s Layer 2 Predicted" % name)
         plt.figure()
         plt.imshow(averages3)
         plt.title("%s Layer 3 Predicted" % name)
         plt.figure()
         plt.imshow(average4)
         plt.title("%s Layer 4 Predicted" % name)
         
    return (average1, average2, average3, average4, compt)
    
    
#def hidden_rep2_GT_fast(net, dataset, device):
#    hidden = [0 for i in range(dataset.branching**dataset.depth)]
#    
#    j = 0
#    test_sequence=[]
#    iterator = [iter(dataset.test_data[k]) for k in range(len(dataset.test_data))]
#    for i in range(len(dataset.test_data)):
#        for j in range(dataset.class_sz_test):
#            test_sequence.append(next(iterator[i]))
#    shuffle(test_sequence)
#    
#    with torch.no_grad():
#        for data in test_sequence:
#            samples, labels = data
#            samples, labels = samples.to(device), labels.to(device)
#            layer2_outputs = net.layer2(samples)
#            if hidden[labels] is 0:
#                hidden[labels] = layer2_outputs
#            else:
#                hidden[labels] = torch.cat((hidden[labels], layer2_outputs), dim=0)
#        for i, data in enumerate(hidden):
#            hidden[i] = torch.mean(data, 0)
#            hidden[i] = hidden[i].numpy()
#    return np.array(hidden)
#            
#
#def hidden_rep1_GT_fast(net, dataset, device):
#    hidden = [0 for i in range(dataset.branching**dataset.depth)]
#    
#    j = 0
#    test_sequence=[]
#    iterator = [iter(dataset.test_data[k]) for k in range(len(dataset.test_data))]
#    for i in range(len(dataset.test_data)):
#        for j in range(dataset.class_sz_test):
#            test_sequence.append(next(iterator[i]))
#    shuffle(test_sequence)
#    
#    with torch.no_grad():
#        for data in test_sequence:
#            samples, labels = data
#            samples, labels = samples.to(device), labels.to(device)
#            layer1_outputs = net.layer1(samples)
#            if hidden[labels] is 0:
#                hidden[labels] = layer1_outputs
#            else:
#                hidden[labels] = torch.cat((hidden[labels], layer1_outputs), dim=0)
#        for i, data in enumerate(hidden):
#            hidden[i] = torch.mean(data, 0)
#            hidden[i] = hidden[i].numpy()
#    return np.array(hidden)
#
#
#def hidden_rep3_GT_fast(net, dataset, device):
#    hidden = [0 for i in range(dataset.branching**dataset.depth)]
#    
#    j = 0
#    test_sequence=[]
#    iterator = [iter(dataset.test_data[k]) for k in range(len(dataset.test_data))]
#    for i in range(len(dataset.test_data)):
#        for j in range(dataset.class_sz_test):
#            test_sequence.append(next(iterator[i]))
#    shuffle(test_sequence)
#    
#    with torch.no_grad():
#        for data in test_sequence:
#            samples, labels = data
#            samples, labels = samples.to(device), labels.to(device)
#            layer3_outputs = net.layer3(samples)
#            if hidden[labels] is 0:
#                hidden[labels] = layer3_outputs
#            else:
#                hidden[labels] = torch.cat((hidden[labels], layer3_outputs), dim=0)
#        for i, data in enumerate(hidden):
#            hidden[i] = torch.mean(data, 0)
#            hidden[i] = hidden[i].numpy()
#    return np.array(hidden)
#
#
#
#
#def plot_hidden_fast(net, dataset, device, name):
#    average1 = hidden_rep1_GT_fast(net, dataset, device)
#    plt.figure()
#    plt.imshow(average1)
#    plt.title("%s Layer 1" % name)
#           
#    average2 = hidden_rep2_GT_fast(net, dataset, device)   
#    plt.figure()
#    plt.imshow(average2)
#    plt.title("%s Layer 2" % name)
#    
#    average3=hidden_rep3_GT_fast(net, dataset, device)
#    plt.figure()
#    plt.imshow(average3)
#    plt.title("%s Layer 3" % name)
#
#




#    for i in range(len(hidden)):
#        hidde
        


#def hidden_rep2_pred(net, dataset, device):
    

#            
#    
#    
#temp3=hidden_rep3_GT(netfc_shuffle, dataset, device)
#average3=avg_hidden_rep(temp3)
#
#average3=np.array(average3)
#plt.figure()
#plt.imshow(average3)
#plt.title('Layer 3')

