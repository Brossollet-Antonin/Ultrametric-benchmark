# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 10:24:07 2020

@author: Simon
"""


import pdb
import itertools
from copy import deepcopy

import random
from random import shuffle
from random import seed

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

import memory
import sort_dataset
import sequence_generator_temporal
import sequence_generator_spatial
import rates_correlation
import preprocessing


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


class Trainer:
    """
    Defined by tuple (dataset, neural_network, sequence_type).
    Training type : random, temporal correlation, spatial correlation, permutedMNIST
    memory sampling : reservoir sampling, ring buffer
    """
    def __init__(self, dataset, network, training_type, memory_sampling, task_sz_nbr=None,
                 sequence_first=0, sequence_length=60000, min_visit=0, energy_step=3, T=1, 
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                 preprocessing=True, proba_transition=1e-3):
        
        self.dataset = dataset
        self.network = network
        self.training_type = training_type

        self.task_sz_nbr = task_sz_nbr
        self.memory_sampling = memory_sampling
        
        self.sequence_first = sequence_first
        self.sequence_length = sequence_length
        self.energy_step = energy_step
        self.T = T
        self.tree_depth = dataset.depth
        self.device = device
        self.min_visit = min_visit
        self.preprocessing = preprocessing
        self.tree_branching = dataset.branching
        self.proba_transition = proba_transition


    def make_train_sequence(self):
        
        if self.training_type=="random":
            j = 0
            train_data=[]
            iterator = [itertools.cycle(self.dataset.train_data[k]) for k in range(len(self.dataset.train_data))]
            for i in range(self.sequence_length):
                if j == len(self.dataset.train_data):
                    j = 0
                train_data.append(next(iterator[j]))
                j += 1
            random.shuffle(train_data)
            self.train_sequence = train_data
        
        elif self.training_type=="temporal correlation":
            seqgen = sequence_generator_temporal.TempCorr_SequenceGenerator()
            train_sequence, rates_vector = seqgen.generate_labels(
                self.sequence_first,
                self.sequence_length,
                self.energy_step,
                self.T,
                self.tree_depth,
                self.tree_branching,
                self.min_visit
            )
            self.train_sequence = (seqgen.generate_data(train_sequence, self.dataset), rates_vector, train_sequence)
            
            
            
        elif self.training_type=="spatial correlation":
            
            data = sort_dataset.sort_MNIST(dataset='MNIST', train=True)
            if self.preprocessing:
                preprocessor = preprocessing.train_preproc()
                rates_matrix = preprocessing.rates_preproc(preprocessor, data, self.T, 10)
            else:
                data = sort_dataset.sort_MNIST(dataset='MNIST', train=True)
                rates_matrix = rates_correlation.rates_cor(data, self.T, 10)

            train_sequence = sequence_generator_spatial.um_sequence_generator(
                self.sequence_first,
                rates_matrix,
                self.sequence_length,
                data,
                minimum_classcount = self.min_visit
            )
            
            self.train_sequence = (sequence_generator_spatial.training_sequence(train_sequence), rates_matrix, train_sequence)


        elif self.training_type=="uniform":
            seqgen = sequence_generator_temporal.Uniform_SequenceGenerator()
            train_sequence, rates_vector = seqgen.uniform_sequence_generator(
                self.sequence_first, self.sequence_length, self.proba_transition, self.tree_depth, self.tree_branching
                )
            self.train_sequence = (seqgen.training_sequence(train_sequence, self.dataset), rates_vector, train_sequence)        
        
        
        elif self.training_type=="onefold split":
            n_classes = self.dataset.branching**self.dataset.depth
            examples_per_class = self.sequence_length // n_classes

            rates = [0 for k in range(self.tree_depth+1)]
            rates[0] = 1

            train_sequence = []
            train_data=[]

            for ex_id in range(n_classes): # MNIST patterns are numbers from 0 to 9
                inst_ids = np.random.randint(0, self.dataset.class_sz_train, size=examples_per_class)
                instances = [self.dataset.train_data[ex_id][inst_ids[k]] for k in range(examples_per_class)]
                train_sequence.extend([ex_id for k in range(examples_per_class)])
                train_data.extend(instances)

            self.train_sequence = (train_data, rates, train_sequence)


        elif self.training_type=="twofold split":
            n_classes = self.dataset.branching**self.dataset.depth
            examples_per_class = self.sequence_length // n_classes

            rates = [0 for k in range(self.tree_depth+1)]
            rates[0] = 0.5
            rates[1] = 0.5

            train_sequence = []
            train_data=[]

            for splt_id in range(n_classes//2): # MNIST patterns are numbers from 0 to 9
                cl_ids = np.random.randint(0, 2, size=2*examples_per_class)
                inst_ids = np.random.randint(0, self.dataset.class_sz_train, size=2*examples_per_class)
                instances = [self.dataset.train_data[2*splt_id+cl_ids[k]][inst_ids[k]] for k in range(2*examples_per_class)]
                # inst = self.dataset.train_data[2*splt_id+cl_ids][inst_ids]
                train_sequence.extend([2*splt_id+cl_ids[k] for k in range(2*examples_per_class)])
                train_data.extend(instances)

            self.train_sequence = (train_data, rates, train_sequence)
            
        else:
            raise NotImplementedError("training type not supported")
            

    def train(self, mem_sz, batch_sz, lr, momentum, training_range):
        if self.training_type in ("temporal correlation", "onefold split", "twofold split", "spatial correlation", "random", "uniform"):
            # For temporal and spatial correlation tests
            n = training_range[0]
            running_loss = 0.0
            
            if self.training_type == 'random':
                train_data = self.train_sequence
            else:
                train_data =  self.train_sequence[0]

            memory_list = [train_data[0]]
            # Define mini-batches of size training.task_sz_nbr and SGD and update the memory for each of them
            while n + self.task_sz_nbr < training_range[1]:
                try:
                    mini_batch = deepcopy(train_data[n])        # train_data[n] is a two elements lists containing tensors
                except:
                    pdb.set_trace()
                for data in train_data[n+1:n+self.task_sz_nbr]:
                    mini_batch[0] = torch.cat((mini_batch[0], data[0]))
                    mini_batch[1] = torch.cat((mini_batch[1], data[1]))
                    
                # Sample elements from memory at random and add it to the mini batch expect for the first iteration     
                if n != 0 and mem_sz != 0:
                    # Sample the memory. We could choose to reduce the number of elements taken from memory, should make things more difficult
                    sample_memory = memory.sample_memory(memory_list, self.task_sz_nbr)
                    train_mini_batch = [torch.cat((mini_batch[0], sample_memory[0])), torch.cat((mini_batch[1], sample_memory[1]))]

                else:
                    train_mini_batch = mini_batch
                # Perform SGD on the mini_batch and memory 
                running_loss += mem_SGD(self.network, train_mini_batch, lr, momentum, self.device)
                # Update memory
                memory_list = memory.reservoir(memory_list, mem_sz, n, mini_batch) if self.memory_sampling == "reservoir sampling" else memory.ring_buffer(memory, mem_sz, n, mini_batch)
                n += self.task_sz_nbr
                if n % (1000*self.task_sz_nbr) == 999*self.task_sz_nbr:
                    print('[%d] loss: %.4f' % (n//self.task_sz_nbr + 1, running_loss/1000))
                    running_loss = 0.0

                

            print("--- Finished Experience Replay training on %s ---" % (training_range,))
       
        else:
            raise NotImplementedError("training type not supported")


    def evaluate(self):
        # Return the accuracy, the predicted and real label for the whole test set and the difference between the two
        # Creation of the testing sequence 
        j = 0
        test_sequence=[]
        iterator = [iter(self.dataset.test_data[k]) for k in range(len(self.dataset.test_data))]
        for i in range(len(self.dataset.test_data)):
            for j in range(self.dataset.class_sz_test):
                test_sequence.append(next(iterator[i]))
        shuffle(test_sequence)
        
        correct = 0
        total = 0
        # Array which will contain the predicted output, the ground truth and the difference of the two
        result = np.zeros((len(test_sequence), 3))
        
        with torch.no_grad():
            for i, data in enumerate(test_sequence):
                samples, labels = data
                samples, labels = samples.to(self.device), labels.to(self.device)
                outputs = self.network(samples)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                result[i][0] = predicted
                result[i][1] = labels
        result[:, 2] = np.abs(result[:, 0] - result[:, 1])
        return (100*correct/total, result)



    def evaluate_hierarchical(self):
        # Return the accuracy, the predicted and GT and the distance between them for every hierachical level 
        # Creation of the testing sequence
        if self.dataset.data_origin=='MNIST' or self.dataset.data_origin=='CIFAR10':
            excluded_labels = [8, 9]
        elif self.dataset.data_origin=='CIFAR100':
            excluded_labels = range(64, 100)
        
        else:
            excluded_labels = []    

        j = 0
        test_sequence=[]
        iterator = [iter(self.dataset.test_data[k]) for k in range(len(self.dataset.test_data) - len(excluded_labels))]   # Create the test sequence with only labels on which the network as been trained on 
        for i in range(len(self.dataset.test_data) - len(excluded_labels)):
            for j in range(self.dataset.class_sz_test):
                test_sequence.append(next(iterator[i]))
        
        shuffle(test_sequence)
        
        # Array which will contain the accuracies at the different hierarchical levels, 
        # the predicted output, the ground truth and the difference of the two
        result = [[], np.zeros((len(test_sequence), 3 + self.tree_depth))]
        with torch.no_grad():
            for i, data in enumerate(test_sequence):
                samples, labels = data
                if labels not in excluded_labels:
                    samples, labels = samples.to(self.device), labels.to(self.device)
                    outputs = self.network(samples)
                    _, predicted = torch.max(outputs.data, 1)
                    result[1][i][0] = predicted
                    result[1][i][1] = labels
        zero = np.zeros(len(test_sequence))
        # Compute the difference between prediction and GT for every hierarchical level
        for i in range(2, self.tree_depth + 3):
            result[1][:, i] = np.abs((result[1][:, 0]//(self.tree_branching**(i-2))) 
                                - (result[1][:, 1]//(self.tree_branching**(i-2))))
            result[0].append((np.sum(result[1][:, i] == zero)/len(test_sequence))*100)
        
        return result

    def testing_final(self):
        # Define which label to exclude in the testing phase (for simplicity, to have a simple tree structure, certain labels have to be excluded
        # depending on the branching and depth of the tree)
        if self.dataset.data_origin=='MNIST' or self.dataset.data_origin=='CIFAR10':
            excluded_labels = [8, 9]
        elif self.dataset.data_origin=='CIFAR100':
            excluded_labels = range(64, 100)
        
        else:
            excluded_labels = []
        # Creation of the testing sequence 
        j = 0
        test_sequence=[]
        iterator = [iter(self.dataset.test_data[k]) for k in range(len(self.dataset.test_data))]
        for i in range(len(self.dataset.test_data)):
            for j in range(self.dataset.class_sz_test):
                test_sequence.append(next(iterator[i]))
        shuffle(test_sequence)
        
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_sequence:
                samples, labels = data
                if labels not in excluded_labels:
                    samples, labels = samples.to(self.device), labels.to(self.device)
                    outputs = self.network(samples)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
        print('Accuracy of the network on the %d test images: %.2f %%' % (total,
        100 * correct / total))
        return (100*correct/total, test_sequence)