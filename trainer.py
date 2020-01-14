# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 13:51:20 2019

@author: Antonin
"""

import torch
from copy import deepcopy
import memory
from train import mem_SGD
import torchvision
import random
import sort_dataset
import sequence_generator_temporal_noself as sequence_generator_temporal
import sequence_generator_spatial
import rates_correlation
import preprocessing
import itertools

import numpy as np

import pdb


class Trainer:
    """
    Define the test that the user wants to perform.
    Training type : random, temporal correlation, spatial correlation, permutedMNIST
    memory sampling : reservoir sampling, ring buffer
    """
    def __init__(self, training_type, memory_sampling, dataset, task_sz_nbr=None,
                 sequence_first=0, sequence_length=60000, train_epoch=None, energy_step=3, T=1, 
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                 preprocessing=True, proba_transition=1e-3):
        self.task_sz_nbr = task_sz_nbr
        self.memory_sampling = memory_sampling
        self.dataset = dataset
        self.training_type = training_type
        self.sequence_first = sequence_first
        self.sequence_length = sequence_length
        self.energy_step = energy_step
        self.T = T
        self.tree_depth = dataset.depth
        self.device = device
        self.train_epoch = train_epoch
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
            return train_data
        
        elif self.training_type=="temporal correlation":
            if self.train_epoch is not None:
                train_sequence, rates_vector = sequence_generator_temporal.um_sequence_generator_epoch(
                    self.sequence_first, self.train_epoch, self.energy_step,
                    self.T, self.tree_depth, self.dataset, self.tree_branching
                    )
            else:
                train_sequence, rates_vector = sequence_generator_temporal.um_sequence_generator(
                    self.sequence_first, self.sequence_length, self.energy_step,
                    self.T, self.tree_depth, self.tree_branching
                    )
            return (sequence_generator_temporal.training_sequence(train_sequence, self.dataset), rates_vector, train_sequence)
            
            
            
        elif self.training_type=="spatial correlation":
            if self.preprocessing:
                if self.train_epoch is not None:
                    data = sort_dataset.sort_dataset(dataset='MNIST', train=True)
                    preprocessor = preprocessing.train_preproc()
                    rates_matrix = preprocessing.rates_preproc(preprocessor, data, self.T, 10)
                    train_sequence = sequence_generator_spatial.um_sequence_generator_epoch(
                        self.sequence_first, rates_matrix, self.train_epoch, data)
                else:
                    data = sort_dataset.sort_MNIST(dataset='MNIST', train=True)
                    preprocessor = preprocessing.train_preproc()
                    rates_matrix = preprocessing.rates_preproc(preprocessor, data, self.T, 10)
                    train_sequence = sequence_generator_spatial.um_sequence_generator(
                        self.sequence_first, rates_matrix, self.sequence_length, data)
            
            else:
                if self.train_epoch is not None:
                    data = sort_dataset.sort_MNIST(dataset='MNIST', train=True)
                    rates_matrix = rates_correlation.rates_cor(data, self.T, 10)
                    train_sequence = sequence_generator_spatial.um_sequence_generator_epoch(
                        self.sequence_first, rates_matrix, self.train_epoch, data)
                else:
                    data = sort_dataset.sort_MNIST(dataset='MNIST', train=True)
                    rates_matrix = rates_correlation.rates_cor(data, self.T, 10)
                    train_sequence = sequence_generator_spatial.um_sequence_generator(
                        self.sequence_first, rates_matrix, self.sequence_length, data)
            return (sequence_generator_spatial.training_sequence(train_sequence), rates_matrix, train_sequence)

        elif self.training_type=="uniform":
            train_sequence, rates_vector = sequence_generator_temporal.uniform_sequence_generator(
                self.sequence_first, self.sequence_length, self.proba_transition, self.tree_depth, self.tree_branching
                )
            return (sequence_generator_temporal.training_sequence(train_sequence, self.dataset), rates_vector, train_sequence)        
        
        
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

            return train_data, rates, train_sequence


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

            return train_data, rates, train_sequence
            

        elif self.training_type=="permutedMNIST":
            pass    #TODO implement permutedMNIST
        else:
            raise NotImplementedError("training type not supported")
            

         
        
        
def train(net, training, train_data_rates, mem_sz, batch_sz, lr, momentum, training_range):
    if training.training_type=="temporal correlation" or training.training_type=="twofold split" or training.training_type=="onefold split" or training.training_type=="spatial correlation" or training.training_type=="random" or training.training_type=="uniform":
        # For temporal and spatial correlation tests
        n = training_range[0]
        running_loss = 0.0
        #train_data_rates=training.train_data()  #Stock rates (if not a random process) and data for training
        train_data =  train_data_rates[0] if training.training_type=="temporal correlation" or training.training_type=="twofold split" or training.training_type=="onefold split" or training.training_type=="spatial correlation" or training.training_type=="uniform" else train_data_rates
#        rates = train_data_rates[1] if training.training_type=="temporal correlation" or training.training_type=="spatial correlation" or training.training_type=="uniform" else 'Random process, no rates involved'
#        train_sequence = train_data_rates[2] if training.training_type=="temporal correlation" or training.training_type=="spatial correlation" or training.training_type=="uniform" else 'Random process, no sequence involved'
        # Initialize the memory with the first example of the serie. Should not really matter. Could also use a random one from the first mini batch
        memory_list = [train_data[0]]
        # Define mini-batches of size training.task_sz_nbr and SGD and update the memory for each of them
        while n + training.task_sz_nbr < training_range[1]:
            mini_batch = deepcopy(train_data[n])        # train_data[n] is a two elements lists containing tensors
            for data in train_data[n+1:n+training.task_sz_nbr]:
                mini_batch[0] = torch.cat((mini_batch[0], data[0]))
                mini_batch[1] = torch.cat((mini_batch[1], data[1]))
                
            # Sample elements from memory at random and add it to the mini batch expect for the first iteration     
            if n != 0 and mem_sz != 0:
                # Sample the memory. We could choose to reduce the number of elements taken from memory, should make things more difficult
                sample_memory = memory.sample_memory(memory_list, training.task_sz_nbr)
                train_mini_batch = [torch.cat((mini_batch[0], sample_memory[0])), torch.cat((mini_batch[1], sample_memory[1]))]

            else:
                train_mini_batch = mini_batch
            # Perform SGD on the mini_batch and memory 
            running_loss += mem_SGD(net, train_mini_batch, lr, momentum, training.device)
            # Update memory
            memory_list = memory.reservoir(memory_list, mem_sz, n, mini_batch) if training.memory_sampling == "reservoir sampling" else memory.ring_buffer(memory, mem_sz, n, mini_batch)
            n += training.task_sz_nbr
            if n % (1000*training.task_sz_nbr) == 999*training.task_sz_nbr:
                print('[%d] loss: %.4f' % (n//training.task_sz_nbr + 1, running_loss/1000))
                running_loss = 0.0
                # Monitor what is inside the memory at a give time
                #print(memory.inspect_memory(memory_list))
            
        # Last mini batch shorter than the requiered length if necessary 
#        mini_batch = deepcopy(train_data[n]) 
#        for data in train_data[n+1:]:
#            mini_batch[0] = torch.cat((mini_batch[0], data[0]))
#            mini_batch[1] = torch.cat((mini_batch[1], data[1]))       # Same number of samples from the memory than from the last mini batch for consistency. Can make an other choice
#        sample_memory = memory.sample_memory(memory_list, mini_batch[0].size(0))
#        train_mini_batch = [torch.cat((mini_batch[0], sample_memory[0])), torch.cat((mini_batch[1], sample_memory[1]))]
#
#        running_loss += mem_SGD(net, train_mini_batch, lr, momentum, training.device)
        print("--- Finished Experience Replay training on %s ---" % (training_range,))
        #return (train_data, rates, train_sequence)

        
    elif training.training_type=="permutedMNIST":
        # TODO for permutedMNIST
        pass
    
    else:
        raise NotImplementedError("training type not supported")