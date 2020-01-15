# -*- coding: utf-8 -*-
"""
Created on Mon May  6 17:42:18 2019

@author: Antonin
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 18:27:21 2019

@author: Antonin
"""


import numpy as np
import random
import itertools
import matplotlib.pyplot as plt
from copy import deepcopy

from statsmodels.tsa.stattools import acf
import pdb

from local_tools import base_conv, verbose, make_ohe

#Set the rates vector
def setting_rates(step, T, tree_depth, branching, rate_law='power', force_switch=True):
    rates = []
    R = np.exp(- step/T)
    for k in range(1, tree_depth+1):
        if rate_law == 'power':
            a = R**k
        elif rate_law == 'exp':
            a = np.exp(-step**k/T)
        elif rate_law == 'log':
            a = np.exp(-np.log(10*k*step)/T)
        else:
            NotImplementedError()
        eps = a/(branching**(k-1))
        rates += (branching**(k-1))*[eps]
    if force_switch:
        rates.insert(0,0) # CAUTION: in this setting, alternation is forced, ie the sequence must change state from one example to the next
    else:
        rates.insert(0,1-sum(rates))
    rates = np.array(rates)
    rates = rates*(1/sum(rates))
    return rates
    

def next_value(sequence, rates, tree_depth, branching):
    i = sequence[-1]
    base_prev = '0'*(tree_depth-len(base_conv(i, branching))) + base_conv(i, branching)
    
    lim = 0
    randomnum = random.random()
    j = 0
    for k,r in enumerate(rates):
        lim += r
        if randomnum <= lim:
            j = k
            break
    indices = (0,0)
    if j == 0:
        indices = (0, 0)
        return int(base_prev,branching)
    elif j in range(1, branching): 
        return int(base_prev[:-1] + str((int(base_prev[-1])+j)%branching), branching)
#        indices = (0, 1)
#        return int(bin_prev[:len(bin_prev)-indices[0]-1] + str((int(bin_prev[len(bin_prev)-indices[0]-1])+1)%2),2)
    for p in range(tree_depth):    
        if j // branching**p == 1 :
            indices = (p, j%(branching**p))
            break

    base_next = base_prev[:len(base_prev)-indices[0]-1] + str((int(base_prev[len(base_prev)-indices[0]-1])+1)%branching) + \
                '0'*(tree_depth - len(base_prev[:len(base_prev)-indices[0]]) - len(base_conv(indices[1], branching))) + base_conv(indices[1], branching)

    return int(base_next, branching)


def sequence_autocor(lbl_sequence, n_labels, nlags=200):
    length = len(lbl_sequence)
    lbl_ohe = make_ohe(lbl_sequence, n_labels)
    autocor = np.zeros(nlags)
    
    for lbl in range(n_labels):
        autocor_lbl = acf(
            lbl_ohe[:,lbl].tolist(),
            unbiased=True,
            nlags=nlags-1, #number of time points to evaluate autocorrelation for
            qstat=False, # allows to return the Ljung-Box q statistic
            fft=True, # this is the fastest method, but impact on accuracy should be assessed when possible
            alpha=None # allows to compute confidence intervals
            )
        autocor = autocor + np.asarray(autocor_lbl)
    return autocor


class SequenceGenerator:
    def __init__(self):
        return

    def generate_labels():
        pass

    def generate_data(self, um_sequence, dataset):
        iterable = [itertools.cycle(dataset.train_data[i]) for i in range(len(dataset.train_data))] 
        train_sequence=[]
        
        for k in um_sequence:
            train_sequence.append(next(iterable[k]))
        return train_sequence



class TempCorr_SequenceGenerator(SequenceGenerator):
    def __init__(self): 
        super().__init__()

    def generate_labels(self, sequence_first, sequence_length, energy_step, T, tree_depth, tree_branching, minimum_classcount = 0, rate_law = 'power', force_switch=True):
        # The following condition is in fact not that necessary to repsect if the energy barrier increase linearly
        #assert (energy_step >= T), 'Unstable stochastic process, Energy_step should be greater than Temperature'
        sequence = [sequence_first]
        rates = setting_rates(energy_step, T, tree_depth, tree_branching, rate_law, force_switch)
        print('Transition rates vector :', rates)
        seq_id = 0

        class_counter = np.array([0 for i in range(2**tree_depth)])
        minvisit_not_satisfied = minimum_classcount

        while (seq_id < sequence_length) or minvisit_not_satisfied:
            next_value_seq = next_value(sequence, rates, tree_depth, tree_branching)
            sequence.append(next_value_seq)
            seq_id += 1
            if minimum_classcount:
                class_counter[next_value_seq] += 1
                minvisit_not_satisfied = ((class_counter < epoch*5000).any())

        return (sequence,rates)



class Uniform_SequenceGenerator(SequenceGenerator):
    def __init__(self): 
        super().__init__()

    def generate_labels(sequence_first, sequence_length, proba_transition, tree_depth, tree_branching):
        sequence = [sequence_first]
        rates = [proba_transition]*(tree_branching**tree_depth - 2)
        assert(sum(rates)<=1), 'Transition probability too high for that many leafs, sum greater than 1. Choose a smaller probability or a smaller tree'
        rates.insert(0, 1-sum(rates))
        rates.insert(0,0)
        print('Transition rates vector :', rates)
        for i in range(sequence_length):
            sequence.append(next_value(sequence, rates, tree_depth, tree_branching))
        return (sequence,rates)


def sequence_autocor_soft(sequence, branching, tree_depth, correlation_type, ratio=2):
    length = len(sequence)
    autocor = []
    
    if correlation_type == 'linear':
        assert tree_depth*ratio*0.1 <= 1, "Ratio too high, pick a smaller ratio (default value is 2)"
        values = [1]
        for i in range(1, tree_depth+1):
            values += [1 - 0.1*ratio*i]*(branching**(i-1))
     
    elif correlation_type == 'exponnential':
        values = [1]
        for i in range(1, tree_depth+1):
            values += [(1/ratio)**i]*(branching**(i-1))
        
    for dt in range(length): 
        sumcor = 0
        for i in range(length - dt):
            sumcor += values[abs(sequence[i] - sequence[i+dt])]
        autocor.append(sumcor/(length - dt))
    
    return autocor

    
def sequence_autocor_fast(sequence, depth, branching, alpha):
    N = len(sequence)
    values = [1]
    for k in range (1, depth+1):
        values += [np.exp(-alpha*k)]*(branching**(k-1))
    seq_val = [values[k] for k in sequence]
    fvi = np.fft.fft(seq_val, n=2*N)
    acf = np.real(np.fft.ifft(fvi*np.conjugate(fvi))[:N])
    acf = acf/N
    print(acf[0])
    return acf  
    

def plot_autocor(T, length, depth, branching, alpha):
    seq = um_sequence_generator(0, length, 1, T, 3,2)[0]
    cor = sequence_autocor_fast(seq, alpha)
    plt.figure()
    plt.plot(seq)
    plt.figure()
    plt.plot(cor)

def plot_autocor_avg(temperature, length, depth, branching, alpha, avg):
    p = 0
    res = np.zeros((len(temperature), length+1))
    fig1 = plt.figure()
    for p,T in enumerate(temperature):
        seq_temp = um_sequence_generator(0, length, 1, T, depth,branching)[0]
        cor = sequence_autocor_fast(seq_temp, depth, branching, alpha)
        plt.figure()
        plt.plot(seq_temp)
        for k in range(avg-1):
            new_cor = sequence_autocor_fast(um_sequence_generator(0, length, 1, T, 
                                                                  depth, branching)[0], 
                                            depth, branching, alpha)
            cor += new_cor
        
        sp = fig1.add_subplot(111)
        sp.plot(cor/avg, label=T)
        res[p,:] = cor/avg 
        p += 1       
    fig1.legend(loc='lower left')
    return res


def sequence_autocor_fixnbr(um_sequence):
    length = len(um_sequence)
    autocor = []
    T = []
    max_val = max(um_sequence)
    lag = [0]+[1.5**i for i in range(1,int(np.log(length)/np.log(1.5))+1)]
    print(len(lag))
    for dt in lag:
        sumcor = 0
        for i in range((length - dt)):
            if um_sequence[i] == um_sequence[i+dt]:
                sumcor += 1
            else:
                sumcor += -1/max_val
        autocor.append(sumcor/(length - dt))
        T.append(dt)
    return [T, autocor]


def sequence_autocor_new(sequence, depth, branching, alpha):
    length = len(sequence)
    autocor = []
    values = [1] #, np.exp(-alpha*1), np.exp(-alpha*2), np.exp(-alpha*2), np.exp(-alpha*3), np.exp(-alpha*3), np.exp(-alpha*3), np.exp(-alpha*3)]
    for k in range(1, depth+1):
        values += [np.exp(-alpha*k) for s in range(branching**(k-1))]
    
    #lag = [0]+[int(1.2**i) for i in range(1,int(np.log(length)/np.log(1.2))+1)]
    lag_float = np.logspace(0, np.log10(length), 100)
    lag = [0] + [int(i) for i in lag_float]
    for dt in lag:
        sumcor = 0
        for i in range(length - dt):
            sumcor += values[abs(sequence[i]-sequence[i+dt])]
        autocor.append(sumcor)
    
    normalize = 1/autocor[0]
    autocor = [a*normalize for a in autocor]
    return [lag, autocor]


def shuffle_sequence(sequence, block_size):
    block_indices = [i for i in range(len(sequence)//block_size)]
    random.shuffle(block_indices)
    shuffled_seq = []
    copied_seq = deepcopy(sequence)
    for i in block_indices:
        shuffled_seq += copied_seq[i*block_size:(i+1)*block_size]
    return shuffled_seq   


def plot_cor():
    random.seed(5)
    depth, ratio = 12, 2
    Temperature = [0.7, 1, 1.2]
    nbr_avg = 1
    length_seq = 1000000
    fig, ax = plt.subplots(figsize=(9,9))
    
    for temp in Temperature:
        seq = um_sequence_generator(0, length_seq, 1, temp, depth, ratio)  
        autocor = np.array(sequence_autocor_new(seq[0], depth, ratio, temp))
        for k in range(nbr_avg-1):
            seq_temp =  um_sequence_generator(0, length_seq, 1, temp, depth, ratio)  
            autocor_temp = np.array(sequence_autocor_new(seq_temp[0], depth, ratio, temp))
            autocor = autocor + autocor_temp
        autocor = autocor/nbr_avg
        plt.loglog(autocor[0], autocor[1])    
        plt.xlabel(r"$\Delta t$", fontsize=22)
        plt.ylabel(r"$A(\Delta t)$", fontsize=22)
        ax.tick_params(labelsize=22)
        
    plt.legend(Temperature, title=r'$T / \Delta$', fontsize=22, title_fontsize=22)
    
    fig, ax = plt.subplots(figsize=(9,9))
    plt.plot(um_sequence_generator(0, length_seq, 1, 1.2, depth, ratio)[0])
    plt.xlabel(r'$t$', fontsize=22)
    plt.ylabel(r"$S(t)$", fontsize=22)
    ax.tick_params(labelsize=22)
    
    fig, ax = plt.subplots(figsize=(9,9))


def plot_cor_block():
    random.seed(5)
    depth, ratio = 9, 2
    temp = 0.7
    nbr_avg = 1
    length_seq = 100000
    Blocks = [1, 100, 10000] + [length_seq]
    fig, ax = plt.subplots(figsize=(9,9)) 
    for block in Blocks:
        seq = um_sequence_generator(0, length_seq, 1, temp, depth, ratio)
        seq_shuffle = shuffle_sequence(seq[0], block)
        autocor = np.array(sequence_autocor_new(seq_shuffle, depth, ratio, temp))
        for k in range(nbr_avg-1):
            seq_temp =  um_sequence_generator(0, length_seq, 1, temp, depth, ratio)
            seq_temp_shuffle = shuffle_sequence(seq_temp[0], block)
            autocor_temp = np.array(sequence_autocor_new(seq_temp_shuffle, depth, ratio, temp))
            autocor = autocor + autocor_temp
        autocor = autocor/nbr_avg
        plt.loglog(autocor[0], autocor[1])    
        plt.xlabel(r"$\Delta t$", fontsize=22)
        plt.ylabel(r"$A(\Delta t)$", fontsize=22)
        ax.tick_params(labelsize=22)
        
    plt.legend(Blocks, title=r'Blocks', fontsize=22, title_fontsize=22)        






