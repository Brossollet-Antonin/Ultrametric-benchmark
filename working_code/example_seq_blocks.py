# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 16:18:29 2019

@author: Antonin
"""

import algo
import torch
import artificial_dataset
import matplotlib.pyplot as plt
import control

dataset = artificial_dataset.artificial_dataset(data_origin='MNIST')

trainer = algo.training('temporal correlation', 'reservoir sampling', dataset=dataset,
                        task_sz_nbr=None,      
                        tree_depth=3, preprocessing=False, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), sequence_length=int(1e6), energy_step=1, T=0.17, 
                        tree_branching=2)
    
train_data_rates=trainer.train_data() 
train_data, rates, train_sequence = train_data_rates[0], train_data_rates[1], train_data_rates[2]


plt.plot(train_sequence)



data_shuffled = control.shuffle_block_partial(train_data, 1000, 400000)
seq=[]
for k in data_shuffled:
    seq.append(k[1].item())
    
plt.figure(figsize=(9,5))
plt.plot(seq, label='Shuffled')
plt.legend()
