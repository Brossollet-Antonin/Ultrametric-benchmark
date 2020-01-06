# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 18:27:21 2019

@author: Antonin
"""


import numpy as np
import random






#Set the rates vector
def setting_rates(step, T, tree_depth, branching):
    rates = []
    R = np.exp(- step/T)
    for k in range(1, tree_depth+1):
        a = R**k
        eps = a/(branching**(k-1))
        rates += (branching**(k-1))*[eps]
    rates.insert(0, 1-sum(rates))
    rates = np.array(rates)
    return rates



def base_conv(value, base):
    # Equivalent to bin(value) but for an arbitrary base. Return a string in the given base
    res = ''
    while value > 0:
        res = str(value % base) + res 
        value = value//base
    return res
    

def next_value(sequence, rates, tree_depth, branching):
    i = sequence[-1]
    base_prev = '0'*(tree_depth-len(base_conv(i, branching))) + base_conv(i, branching)
    
    lim = 0
    randomnum = random.random()
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
    
def um_sequence_generator(sequence_first, sequence_length, energy_step, T, tree_depth, tree_branching):
    # The following condition is in fact not that necessary to repsect if the energy barrier increase linearly
    #assert (energy_step >= T), 'Unstable stochastic process, Energy_step should be greater than Temperature'
    sequence = [sequence_first]
    rates = setting_rates(energy_step, T, tree_depth, tree_branching)
    print('Transition rates vector :', rates)
    for i in range(sequence_length):
        sequence.append(next_value(sequence, rates, tree_depth, tree_branching))
    return (sequence,rates)


def um_sequence_generator_epoch(sequence_first, epoch, energy_step, T, tree_depth, dataset, tree_branching):
    # The following condition is in fact not that necessary to repsect if the energy barrier increase linearly    
    #assert (energy_step >= T), 'Unstable stochastic process, Energy_step should be greater than Temperature'
    sequence = [sequence_first]
    rates = setting_rates(energy_step, T, tree_depth, tree_branching)
    print('Transition rates vector :', rates)
    compteur = np.array([0 for i in range(2**tree_depth)])
    while (compteur < epoch*5000).any():   # Make sure that we see each classes a certain number of times. # TODO: change it to a non hardcoded version
        next_value_seq = next_value(sequence, rates, tree_depth, tree_branching)
        sequence.append(next_value_seq)
        compteur[next_value_seq] += 1
    return (sequence,rates)


def training_sequence(um_sequence, dataset):
    iterable = [itertools.cycle(dataset.train_data[i]) for i in range(len(dataset.train_data))] 
    train_sequence=[]
    for k in um_sequence:
        train_sequence.append(next(iterable[k]))
    return train_sequence
  
    
def sequence_autocor(um_sequence):
    length = len(um_sequence)
    autocor = []
    max_val = max(um_sequence)
    for dt in range(length):
        sumcor = 0
        for i in range(length - dt):
            if um_sequence[i] == um_sequence[i+dt]:
                sumcor += 1
            else:
                sumcor += -1/max_val
        autocor.append(sumcor/(length - dt))
    return autocor


def uniform_sequence_generator(sequence_first, sequence_length, proba_transition, tree_depth, tree_branching):
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



#seq = um_sequence_generator(0, 10000,1, 1/3, 3, 2)[0]
#autocor_linear = sequence_autocor_soft(seq[:10000], 2, 3, 'linear')
#plt.plot(autocor_linear)
#plt.figure()
#plt.plot(seq)


#T=1
#step=5
#um_sequence=um_sequence_generator(0,5000000,step,T)
#autocor=sequence_autocor(um_sequence)
#
##
##
##t=np.linspace(0.1,10,5000001)
##y=np.exp(-t)/t
##z=np.exp(-(T*np.log(2)/step)*np.log(t))
##plt.plot(t,y)
##plt.plot(t,z)
##plt.plot(t,autocor)
#
#random_sequence=[0]+[random.randint(0,7) for i in range(5000000)]
#autocor_random=sequence_autocor(random_sequence)
#plt.plot(autocor_random, label='Random')
#plt.plot(autocor, label='Ultrametric')
#plt.legend()
#autocorrelation= signal.correlate(um_sequence,um_sequence)
#
#


#
#um_sequence=um_sequence_generator(5, sequence_length, 3,1)
#resultat=training_sequence(um_sequence)
#resultat[0].size()
#
#sequence=um_sequence_generator(5, sequence_length, 3, 1)
#
#compteur=8*[0]
#for k in sequence:
#    compteur[k]+=1
#    
#compteur2=8*[0]
#for k in sequence:
#    compteur2[k]+=1
#
#    
# # Marche très bien pour pouvoir iterer une fois que j'aurai récupérer la scéance de chiffres
## que je veux générer
#iterable=[iter(train_data_sorted[0]), iter(train_data_sorted[1]), iter(train_data_sorted[2])]
#
#test=next(iterable[0])
#print(test)
#
#binaire=list(bin(6))
#binaire[4]= str((int(binaire[4])+1)%2)
#print(int((''.join(binaire)[2:])))
#
#
#i=0
#
#p=2
#i2= int(bin(i)[2:len(bin(i))-p-1] + bin(toggleBit(i, p ))[len(bin(i))-p-1] + bin(i%(2**p))[2:], 2)
#
#
#
#for i in range(200,300):
#    print(bin(i)[len(bin(i))-p-1]== bin(toggleBit(i, p))[len(bin(i))-p-1])
#
#    
    