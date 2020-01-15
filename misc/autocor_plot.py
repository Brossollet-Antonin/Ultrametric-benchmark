# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 15:14:17 2019

@author: Antonin
"""

import sequence_generator_temporal as sequence_generator
#import matplotlib.pyplot as plt
import numpy as np
import control


X0 = [0,10000]
Y0 = [0,0]
plt.figure()
plt.plot(X0, Y0)

Temperature = [1/1.2, 1/1.4, 1/1.8, 1/2, 1/2.2, 1.5, 2, 1/3]
cor = [[] for i in range(len(Temperature))]
sequence = [[] for i in range(len(Temperature))]

for i,T in enumerate(Temperature):
    sequence[i] = sequence_generator.um_sequence_generator(0, 10000, 1, T, 7, 2)[0]
    cor[i] = sequence_generator.sequence_autocor(sequence[i])
    plt.plot(cor[i], label=T)
    
plt.legend()

plt.figure()
plt.plot(sequence[0])
plt.figure()
plt.plot(sequence[5])    
plt.figure()
plt.plot(sequence[2])





average = 9
T=1/4
cor_avg =  np.array(sequence_generator.sequence_autocor(sequence_generator.um_sequence_generator(0, 10000, 1, T, 3, 2)[0]))
for i in range(average):
    sequence_ = np.array(sequence_generator.um_sequence_generator(0, 10000, 1, T, 3, 2)[0])
    cor_avg = cor_avg + sequence_generator.sequence_autocor(sequence_)
    
cor_avg = cor_avg/10

plt.figure()

plt.plot(cor_avg)


average = 9
T=1/4
block_size = 500
cor_avg_block =  np.array(sequence_generator.sequence_autocor(control.shuffle_block(sequence_generator.um_sequence_generator(0, 10000, 1, T, 3, 2)[0], block_size)))
for i in range(average):
    sequence_ = np.array(control.shuffle_block(sequence_generator.um_sequence_generator(0, 10000, 1, T, 3, 2)[0], block_size))
    cor_avg_block = cor_avg_block + sequence_generator.sequence_autocor(sequence_)
    
cor_avg_block = cor_avg_block/10

plt.figure()
plt.plot(cor_avg_block)