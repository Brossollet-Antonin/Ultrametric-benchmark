# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 16:09:01 2019

@author: Antonin
"""


import random 

from copy import deepcopy
import numpy as np
import memory
import torch
import sequence_generator_temporal
from operator import add

import pdb
from trainer import mem_SGD

 
def generate_batch(train_sequence, itr, batch_sz, first_train_id):
        train_labels = train_sequence[first_train_id:first_train_id+batch_sz]
        first_couple = next(itr[train_labels[0]])
        train_data = first_couple[0]
        train_tensorlabels = first_couple[1]
        for seq_locid in range(1,batch_sz):
            next_couple = next(itr[train_labels[seq_locid]])
            train_data = torch.cat((train_data, next_couple[0]))
            train_tensorlabels = torch.cat((train_tensorlabels, next_couple[1]))
        
        return [train_data, train_tensorlabels]

    
def train(net, training, control_labels, mem_sz, batch_sz, lr, momentum, training_range):
    """
    Train a network on the specified training protocol.

    Parameters
    ----------
    net : neural network
        The neural network to train.
    training  
        Trainer object parametrizing the training protocol.
    control_data
        Data used to do the training.
    mem_sz : int
        Size of the memory.
    batch_sz : int
        Size of the mini-batches for the training.
    lr : int
        Learning rate.
    momentum : int
        Momentum for the SGD.
    training_range : list
        Range of the data to train on. training_range[0] is the begining 
        sample, training_range[1] is the end sample

    Returns
    -------
    None.

    """
    # THIS IS ACTUALLY learning_ER with a shuffled sequence.
    # ToDo: either rename this method to clarify or factorize the code to have a single method
    #if training.training_type=="temporal correlation" or training.training_type=="spatial correlation" or training.training_type=="random" or tranini:
    # For temporal and spatial correlation tests
    
    first_train_id = training_range[0]
    running_loss = 0.0
#    control_data =  deepcopy(train_data)
#    random.shuffle(control_data)
#    length_data = len(control_data)        # maybe torch.size(0) if we stock the data in a tensor...
    # Initialize the memory with the first example of the serie. Should not really matter. Could also use a random one from the first mini batch
    memory_list = [next(training.data_iterator[control_labels[0]])]
    # Define mini-batches of size training.batch_sz and SGD and update the memory for each of them
    while first_train_id + training.batch_sz < training_range[1]:
        mini_batch = generate_batch(control_labels, training.data_iterator, training.batch_sz, first_train_id) # control_data[n] is a two elements lists containing tensors
            
        # Sample elements from memory at random and add it to the mini batch expect for the first iteration     
        if first_train_id != 0:
            sample_memory = memory.sample_memory(memory_list, training.batch_sz)
            train_mini_batch = [
                torch.cat((mini_batch[0], sample_memory[0])),
                torch.cat((mini_batch[1], sample_memory[1]))
            ]

        else:
            train_mini_batch = mini_batch
        # Perform SGD on the mini_batch and memory 
        running_loss += mem_SGD(net, train_mini_batch, lr, momentum, training.device)
        # Update memory
        memory_list = memory.reservoir(memory_list, mem_sz, first_train_id, mini_batch) if training.memory_sampling == "reservoir sampling" else memory.ring_buffer(memory, mem_sz, first_train_id, mini_batch)
        first_train_id += training.batch_sz
        if first_train_id % (1000*training.batch_sz) == 999*training.batch_sz:
            print('[%d] loss: %.4f' % (first_train_id//training.batch_sz + 1, running_loss/1000))
            running_loss = 0.0
        
#    # Last mini batch shorter than the requiered length if necessary 
#    mini_batch = deepcopy(control_data[n]) 
#    for data in control_data[n+1:]:
#        mini_batch[0] = torch.cat((mini_batch[0], data[0]))
#        mini_batch[1] = torch.cat((mini_batch[1], data[1]))       # Same number of samples from the memory than from the last mini batch for consistency. Can make an other choice
#    sample_memory = memory.sample_memory(memory_list, mini_batch[0].size(0))
#    train_mini_batch = [torch.cat((mini_batch[0], sample_memory[0])), torch.cat((mini_batch[1], sample_memory[1]))]
#
#    running_loss += train.mem_SGD(net, train_mini_batch, lr, momentum, training.device)
    
    #Testing the newly trained NN on the testset
    
    print("Control on shuffled sequence %s" % (training_range,))
#    accuracy, _ = testing.testing_final(net, training.dataset, training.device)
#    return (control_data, accuracy) 


def shuffle_block(train_data, block_size):
    """
    Block shuffle a data sequence.

    Parameters
    ----------
    train_data : list of torch.Tensor
        List of the ordered samples used in the training process.
    block_size : int
        Size of the block for the shuffle.

    Returns
    -------
    control_data_block : list of torch.Tensor
        List of the samples shuffled by blocks.

    """
    # Return a sequence shuffled by block to see the effect of a cut off in the long scale temporal correlation
    block_indices = [i for i in range(len(train_data)//block_size)]
    random.shuffle(block_indices)
    control_data_block = []
    copied_train_data = deepcopy(train_data)
    for i in block_indices:
        control_data_block += copied_train_data[i*block_size:(i+1)*block_size]
    return control_data_block
    

def shuffle_block_stef(train_data, block_size):
    """
    Block shuffle a data sequence using Stefano & Marcus method.

    Parameters
    ----------
    train_data : list of torch.Tensor
        List of the ordered samples used in the training process.
    block_size : int
        Size of the block for the shuffle.

    Returns
    -------
    control_data_block : list of torch.Tensor
        List of the samples shuffled by blocks.

    """    
    nb = len(train_data)//block_size    #number of blocks
    ns = 10*nb      #number of shuffles that will be done
    train_data_shuffle = deepcopy(train_data)
    for i in range(ns):
        ia = random.randint(0, nb-1)*block_size
        ib = random.randint(0, nb-1)*block_size
        while ia == ib:
            ib = random.randint(0, nb-1)*block_size
        databuffer = train_data_shuffle[ia:ia+block_size]
        train_data_shuffle[ia:ia+block_size] = train_data_shuffle[ib:ib+block_size]
        train_data_shuffle[ib:ib+block_size] = databuffer
    return train_data_shuffle


def shuffle_block_partial(train_labels, block_size, end): 
    """
    Block shuffle a data sequence up to a certain point.
    
    The part of the sequence aft

    Parameters
    ----------
    train_data : list of torch.Tensor
        List of the ordered samples used in the training process.
    block_size : int
        Size of the block for the shuffle.
    end: int 
        Indice of the last sample of the sequence to shuffle 
    
    Returns
    -------
    control_data_block : list of torch.Tensor
        List of the samples shuffled by blocks.

    """    
    block_indices = list(range(len(train_labels[:end])//block_size))
    random.shuffle(block_indices)
    block_indices += list(range(len(train_labels[:end])//block_size, len(train_labels)//block_size))     #add the rest of the data unshuffled to have everything work smoothly with older code. Not optimal but simpler
    #shuffled_data = []
    #shuffled_labels = []
    #copied_train_data = deepcopy(train_data)
    #copied_train_labels = deepcopy(train_labels)
    idx_shuffled = list(
        map(
            add,
            np.repeat([block_size*k for k in block_indices], block_size).tolist(),
            list(range(block_size))*len(block_indices)
        )
    )

    #for i in block_indices:
    #    shuffled_data += copied_train_data[i*block_size:(i+1)*block_size]
    #    shuffled_labels += copied_train_labels[i*block_size:(i+1)*block_size]

    shuffled_labels = [train_labels[k] for k in idx_shuffled]

    return shuffled_labels

    
    
    
    
        
def shuffle_labels(net, training, control_data, mem_sz, batch_sz, lr, momentum, training_range):
    
#    permutation = [i for i in range(max(train_sequence)+1)]
#    random.shuffle(permutation)
#    control_sequence = [permutation[i] for i in train_sequence]
#    # Generate a sequence similar to the one use for the training, but with permuted labels
#    control_data = sequence_generator_temporal.training_sequence(control_sequence, training.dataset)
    n = training_range[0]
    running_loss = 0.0
    length_data = len(control_data)        # maybe torch.size(0) if we stock the data in a tensor...
    # Initialize the memory with the first example of the serie. Should not really matter. Could also use a random one from the first mini batch
    memory_list = [control_data[0]]
    # Define mini-batches of size training.task_sz_nbr and SGD and update the memory for each of them
    while n + training.task_sz_nbr < training_range[1]:
        mini_batch = deepcopy(control_data[n])        # control_data[n] is a two elements lists containing tensors
        for data in control_data[n+1:n+training.task_sz_nbr]:
            mini_batch[0] = torch.cat((mini_batch[0], data[0]))
            mini_batch[1] = torch.cat((mini_batch[1], data[1]))
            
        # Sample elements from memory at random and add it to the mini batch expect for the first iteration     
        if n != 0:
            sample_memory = memory.sample_memory(memory_list, training.task_sz_nbr)
            train_mini_batch = [torch.cat((mini_batch[0], sample_memory[0])), torch.cat((mini_batch[1], sample_memory[1]))]

        else:
            train_mini_batch = mini_batch
        # Perform SGD on the mini_batch and memory 
        running_loss += train.mem_SGD(net, train_mini_batch, lr, momentum, training.device)
        # Update memory
        memory_list = memory.reservoir(memory_list, mem_sz, n, mini_batch) if training.memory_sampling == "reservoir sampling" else memory.ring_buffer(memory, mem_sz, n, mini_batch)
        n += training.task_sz_nbr
        if n % (1000*training.task_sz_nbr) == 999*training.task_sz_nbr:
            print('[%d] loss: %.4f' % (n//training.task_sz_nbr + 1, running_loss/1000))
            running_loss = 0.0
        
    # Last mini batch shorter than the requiered length if necessary 
#    mini_batch = deepcopy(control_data[n]) 
#    for data in control_data[n+1:]:
#        mini_batch[0] = torch.cat((mini_batch[0], data[0]))
#        mini_batch[1] = torch.cat((mini_batch[1], data[1]))       # Same number of samples from the memory than from the last mini batch for consistency. Can make an other choice
#    sample_memory = memory.sample_memory(memory_list, mini_batch[0].size(0))
#    train_mini_batch = [torch.cat((mini_batch[0], sample_memory[0])), torch.cat((mini_batch[1], sample_memory[1]))]
#
#    running_loss += train.mem_SGD(net, train_mini_batch, lr, momentum, training.device)
    
    #Testing the newly trained NN on the testset

    print("Control on sequence with shuffled labels %s" % (training_range,))
#    accuracy, _ = testing.testing_final(net, training.dataset, training.device)
    #return (control_data, accuracy)
    
    


        
        
        
        
#def shuffle_block_old(train_data, block_size):
## Return a sequence shuffled by block to see the effect of a cut off in the long scale temporal correlation
#block_indices = [i for i in range(len(train_data)//block_size)]
#random.shuffle(block_indices)
#control_data_block = [[] for i in range(len(train_data))]
#copied_train_data = deepcopy(train_data)
#for j, i in enumerate(block_indices):
#    control_data_block[i*block_size:(i+1)*block_size] = copied_train_data[j*block_size:(j+1)*block_size]
#return control_data_block    
#    
        
        
        
        
        
        