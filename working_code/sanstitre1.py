# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 16:05:50 2019

@author: Antonin
"""

import control
import sequence_generator_temporal_noself as sege

import matplotlib.pyplot as plt
import random

random.seed(6432165)

seq1=sege.um_sequence_generator(0,1000000,1,0.170,3,2)[0]
fig, ax = plt.subplots(figsize=(9,9))
ax.tick_params(labelsize=22)
plt.plot(seq1)
ax.set_xlabel('Number of training samples', fontsize=22)
ax.set_ylabel('Label', fontsize=22)

seq4=control.shuffle_block(seq1, 1000)
fig, ax = plt.subplots(figsize=(9,9))
ax.tick_params(labelsize=22)
plt.plot(seq4)
ax.set_xlabel('Number of training samples', fontsize=22)
ax.set_ylabel('Label', fontsize=22)

seq2=control.shuffle_block(seq1, 10000)
fig, ax = plt.subplots(figsize=(9,9))
ax.tick_params(labelsize=22)
plt.plot(seq2)
ax.set_xlabel('Number of training samples', fontsize=22)
ax.set_ylabel('Label', fontsize=22)

seq3=control.shuffle_block(seq1, 100000)
fig, ax = plt.subplots(figsize=(9,9))
ax.tick_params(labelsize=22)
plt.plot(seq3)
ax.set_xlabel('Number of training samples', fontsize=22)
ax.set_ylabel('Label', fontsize=22)

