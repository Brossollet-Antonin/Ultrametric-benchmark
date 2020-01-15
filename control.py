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

    
    
def shuffle_sequence(net, training, control_data, mem_sz, batch_sz, lr, momentum, training_range):
    # THIS IS ACTUALLY learning_ER with a shuffled sequence.
    # ToDo: either rename this method to clarify or factorize the code to have a single method
    #if training.training_type=="temporal correlation" or training.training_type=="spatial correlation" or training.training_type=="random" or tranini:
    # For temporal and spatial correlation tests
    
    n = training_range[0]
    running_loss = 0.0
#    control_data =  deepcopy(train_data)
#    random.shuffle(control_data)
#    length_data = len(control_data)        # maybe torch.size(0) if we stock the data in a tensor...
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


#def shuffle_block2(train_data, block_size):
#    # Return a sequence shuffled by block to see the effect of a cut off in the long scale temporal correlation
#    block_indices = [i for i in range(len(train_data)//block_size)]
#    random.shuffle(block_indices)
#    control_data_block = []
#    copied_train_data = deepcopy(train_data)
#    for i in block_indices:
#        control_data_block = control_data_block + copied_train_data[i*block_size:(i+1)*block_size]
#    return control_data_block




def shuffle_block(train_data, block_size):
    # Return a sequence shuffled by block to see the effect of a cut off in the long scale temporal correlation
    block_indices = [i for i in range(len(train_data)//block_size)]
    random.shuffle(block_indices)
    control_data_block = []
    copied_train_data = deepcopy(train_data)
    for i in block_indices:
        control_data_block += copied_train_data[i*block_size:(i+1)*block_size]
    return control_data_block
    

def shuffle_block_stef(train_data, block_size):
    # Return a sequence shuffled by block using the different method from Stefano and Marcus for the shuffle 
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


def shuffle_block_partial(train_dataset, block_size, end): 
    # Shuffle the sequence by blocks up to a chosen ending point
    train_data, rates, train_labels = train_dataset
    block_indices = [i for i in range(len(train_data[:end])//block_size)]
    random.shuffle(block_indices)
    block_indices += [i for i in range(len(train_data[:end])//block_size, len(train_data)//block_size)]     #add the rest of the data unshuffled to have everything work smoothly with older code. Not optimal but simpler
    shuffled_data = []
    shuffled_labels = []
    copied_train_data = deepcopy(train_data)
    copied_train_labels = deepcopy(train_labels)
    for i in block_indices:
        shuffled_data += copied_train_data[i*block_size:(i+1)*block_size]
        shuffled_labels += copied_train_labels[i*block_size:(i+1)*block_size]
    return shuffled_data, rates, shuffled_labels

    
    
    
    
        
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
        
        
        
        
        
        