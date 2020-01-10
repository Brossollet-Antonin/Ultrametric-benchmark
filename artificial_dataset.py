# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 16:09:45 2019

@author: Antonin
"""


import torch
import random
from copy import deepcopy
import sort_dataset



class artificial_dataset:
    
    def __init__(self, data_origin, depth=0, branching=0, data_sz=0, class_sz_train=0, class_sz_test=0, ratio_type='linear', ratio_value=1, noise_level=1):
        self.data_origin = data_origin
        self.depth = depth
        self.class_sz_train = class_sz_train
        self.branching = branching
        self.data_sz = data_sz
        self.class_sz_test = class_sz_test
        self.ratio_type = ratio_type
        self.ratio_value = ratio_value
        self.noise_level = noise_level
        if data_origin != 'artificial':
            self.train_data = sort_dataset.sort_dataset(dataset=data_origin, train=True)
            self.test_data = sort_dataset.sort_dataset(dataset=data_origin, train=False)
            if data_origin=='MNIST':
                self.class_sz_test = 892    # The class 5 of MNIST as only 892 test samples
            elif data_origin=='CIFAR10':
                self.class_sz_test = 1000
            else: 
                self.class_sz_test = 100
        else:
            self.create()
        
    
    def create(self):
        # Creating an initial random tensor of size data_sz, of +1 and -1
        initial = torch.randint(2,(self.data_sz,), dtype=torch.float)
        initial = initial*2.0 -1.0
        
        # List to stock the parents to allow the creation of the different
        parents_mem = [initial]
        train_data = [[] for i in range(self.branching**self.depth)]
        test_data = [[] for i in range(self.branching**self.depth)]
        label = 0
        d = 0
        b = 0
        while label < self.branching**self.depth:
            if b == self.branching:
                parents_mem.pop()
                b = 0
                d -= 1
            next_parent = deepcopy(parents_mem[-1])
            if self.ratio_type == 'linear':
                for s in range(self.ratio_value):             
                    ind_mod = random.randint(0, self.data_sz -1)
                    next_parent[ind_mod] = next_parent[ind_mod]*(-1)
            elif self.ratio_type == 'exponnential': 
                for s in range(int(self.data_sz*(1/self.ratio_value)**(d+1))):
                    ind_mod = random.randint(0, self.data_sz -1)
                    next_parent[ind_mod] = next_parent[ind_mod]*(-1)                    
            else:
                raise NotImplementedError("Supported modes are for the moment 'linear' and 'exponnential'")
            # Switch the value of the ind_mod digit
            
            parents_mem.append(next_parent) 
            d += 1
            # If at the bottom of the tree, create the random samples
            if d == self.depth:
                b += 1
                # Creating train set
                for k in range(self.class_sz_train):
                    new_example_train = deepcopy(parents_mem[-1])
                    # Add noise
                    for s in range(self.noise_level):
                        ind_mod_ex_train = random.randint(0, self.data_sz -1)
                        new_example_train[ind_mod_ex_train] = new_example_train[ind_mod_ex_train]*(-1)
                    # Add mini batch size and number of channel to have a correctly formated tensor for training
                    new_example_train = torch.unsqueeze(new_example_train, 0)
                    new_example_train = torch.unsqueeze(new_example_train, 0)
                    train_data[label].append([new_example_train, torch.tensor([label])])
                # Creating test set
                for k in range(self.class_sz_test):
                    new_example_test = deepcopy(parents_mem[-1])
                    # Add noise
                    for s in range(self.noise_level):
                        ind_mod_ex_test = random.randint(0, self.data_sz -1)
                        new_example_test[ind_mod_ex_test] = new_example_test[ind_mod_ex_test]*(-1)
                    # Add mini batch size and number of channel to have a correctly formated tensor for training
                    new_example_test = torch.unsqueeze(new_example_test, 0)
                    new_example_test = torch.unsqueeze(new_example_test, 0)
                    test_data[label].append([new_example_test, torch.tensor([label])])
                    
                label += 1 
                d -= 1
            
        self.train_data = train_data
        self.test_data = test_data
                
         
#artific=artificial_dataset(4,20,3,150)
#data=artific.create()     
#
#
#
#
#import rates_correlation
#cor_matrix=rates_correlation.avg_cor(data, 5)

       


                
    