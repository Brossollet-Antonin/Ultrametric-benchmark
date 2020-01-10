# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 15:57:29 2019

@author: Antonin
"""

import random
import torch
from copy import deepcopy


def sample_memory(memory, sample_sz):
    '''Return a random tensor sample from the memory of size sample_sz'''
    # Random indices of the memory which will be selected
    random_indices = [random.randint(0, len(memory)-1) for k in range(sample_sz)]
    sample_mem = deepcopy(memory[random_indices[0]])
    for k,ind in enumerate(random_indices[1:]):
        sample_mem[0] = torch.cat((sample_mem[0], memory[ind][0]))
        if sample_mem[1].size() != torch.Size([]):
            if memory[ind][1].size() == torch.Size([]):
                sample_mem[1] = torch.cat((sample_mem[1], torch.unsqueeze(memory[ind][1],0)))
            else:
                sample_mem[1] = torch.cat((sample_mem[1], memory[ind][1]))
        else:
            if memory[ind][1].size() == torch.Size([]):
                sample_mem[1] = torch.cat((torch.unsqueeze(sample_mem[1],0), torch.unsqueeze(memory[ind][1],0)))
            else:
                sample_mem[1] = torch.cat((torch.unsqueeze(sample_mem[1],0), memory[ind][1]))
            
    return sample_mem



def sample_memory_ringbuffer(memory, sample_sz):
    '''Return a random tensor sample from the memory of size sample_sz'''
    # Random indices of the memory which will be selected
    random_labels = [random.randint(0,10) for k in range(sample_sz)]
    random_indices = [random.randint(0, len(memory[0])-1) for k in range(sample_sz)]
    sample_mem = deepcopy(memory[random_labels[0]][random_indices[0]])
    for k,ind in enumerate(random_labels[1:]):
        sample_mem[0] = torch.cat((sample_mem[0], memory[ind][0]))
        if sample_mem[1].size() != torch.Size([]):
            if memory[ind][1].size() == torch.Size([]):
                sample_mem[1] = torch.cat((sample_mem[1], torch.unsqueeze(memory[ind][1],0)))
            else:
                sample_mem[1] = torch.cat((sample_mem[1], memory[ind][1]))
        else:
            if memory[ind][1].size() == torch.Size([]):
                sample_mem[1] = torch.cat((torch.unsqueeze(sample_mem[1],0), torch.unsqueeze(memory[ind][1],0)))
            else:
                sample_mem[1] = torch.cat((torch.unsqueeze(sample_mem[1],0), memory[ind][1]))
            
    return sample_mem


def reservoir(memory, mem_sz, n, mini_batch):
    '''Update the memory using reservoir sampling'''
    j = 0
    for k in range(mini_batch[0].size(0)):
        M = len(memory)
        if M < mem_sz:
            memory.append([torch.unsqueeze(mini_batch[0][k],0), mini_batch[1][k]])  # Add a list [image (tensor), label (tensor)] to the memory
        else:
            i = random.randint(0, n+j)
            if i < mem_sz:
                memory[i] = deepcopy([torch.unsqueeze(mini_batch[0][k],0), mini_batch[1][k]])
        j += 1
    return memory
            
    
    
    
# see for FIFO stacks of fixed length: https://stackoverflow.com/questions/1931589/python-datatype-for-a-fixed-length-fifo
# Memory is a list of size the number of labels and for each label there is a deque stack containing the sample for this label
def ring_buffer(memory, mini_batch):
    #TODO
    ''' Update the memory using ring buffer. Stacks is a list with one FIFO stack for each label'''
    for k in range(mini_batch[0].size(0)):
        # Get the label to know in which stack we need to put the sample
        label = mini_batch[1][k]
        memory[label].appendleft([torch.unsqueeze(mini_batch[0][k],0), mini_batch[1][k]])
    return memory
        



# To inspect what labels are represented in the memory at a give moment        
def inspect_memory(memory): 
    compteur = [0 for i in range(10)]
    for data in memory: 
        label = data[1].item()
        compteur[label] += 1
    return compteur
        
        
        
        
        
        
        
        
        
        
    