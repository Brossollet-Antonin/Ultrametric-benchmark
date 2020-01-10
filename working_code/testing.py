# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 15:48:51 2019

@author: Antonin
"""

import torch
from random import shuffle
#import matplotlib.pyplot as plt

def testing_final(net, dataset, device):
    
    # Define which label to exclude in the testing phase (for simplicity, to have a simple tree structure, certain labels have to be excluded
    # depending on the branching and depth of the tree)
    if dataset.data_origin=='MNIST' or dataset.data_origin=='CIFAR10':
        excluded_labels = [8, 9]
    elif dataset.data_origin=='CIFAR100':
        excluded_labels = range(64, 100)
    
    else:
        excluded_labels = []
    # Creation of the testing sequence 
    j = 0
    test_sequence=[]
    iterator = [iter(dataset.test_data[k]) for k in range(len(dataset.test_data))]
    for i in range(len(dataset.test_data)):
        for j in range(dataset.class_sz_test):
            test_sequence.append(next(iterator[i]))
    shuffle(test_sequence)
    
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_sequence:
            samples, labels = data
            if labels not in excluded_labels:
                samples, labels = samples.to(device), labels.to(device)
                outputs = net(samples)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the %d test images: %.2f %%' % (total,
    100 * correct / total))
    return (100*correct/total, test_sequence)
    
    

#def testing_final(net, test_data, device, excluded_labels):
#    correct = 0
#    total = 0
#    total_prediction=[0]
#    with torch.no_grad():
#        for data in test_data:
#            images, labels = data
#            if labels not in excluded_labels:
#                images, labels = images.to(device), labels.to(device)
#                outputs = net(images)
#                _, predicted = torch.max(outputs.data, 1)
#                total += labels.size(0)
#                correct += (predicted == labels).sum().item()
#    #            print(labels, predicted)
#                #see if right answer are evenly distributed
#                total_sum=total_prediction[-1] + (predicted[0] == labels[0]).item()
#                total_prediction.append(total_sum)
#                
#    
#    print('Accuracy of the network on the %d test images: %d %%' % (total,
#        100 * correct / total))
#    plt.figure()
#    plt.plot(total_prediction)
#    print(total_prediction)
#    return 100*correct/total 