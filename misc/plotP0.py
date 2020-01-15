# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 17:38:03 2019

@author: Antonin
"""

import pickle
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import random
import sequence_generator_temporal as sequence_generator


tree_l = 2**6
maxh = 2**19    
blocks = [1, 10, 100, 1000]
nbrtest = 10
loadpath = "/Users/Antonin/Documents/Documents/ENS 2A/Stage M1/Results/Blocks/MNIST/NewShuffle/block1 T0.150/3/"

original = pickle.load(open(loadpath+'original','rb'))
#shuffled = pickle.load(open(loadpath+'shuffle','rb'))
#
#
#
#for block in blocks:
#    plt.figure()
#    for seqname in ['original', 'shuffle']:
#        seq = pickle.load(open(loadpath+seqname,'rb'))
#        hlocs_stat = np.zeros(maxh-1)
#        for i in range(tree_l):
#            locs = np.array([j for j in range(len(seq)) if seq[j]==i])
#            locss = deepcopy(locs)
#            locss[:-1] = locss[1:]
#            locsd = locss-locs
#            bins = range(maxh)
#            hlocs = np.histogram(locsd, bins, density=True)
#            hlocs_stat = hlocs_stat + hlocs[0]/tree_l
# 
#        plt.loglog(bins[:-1], hlocs_stat, label=seqname) 
#        plt.title(block)
#        plt.legend()

 
    
def shuffleblocks(seq, block_sz, snbr):
    lseq = len(seq)
    copied_seq = deepcopy(seq)
    sseq = []   # Will contain the shuffled sequence
    for k in range(snbr):
        begin, end = int(k*lseq/snbr), int((k+1)*lseq/snbr)
        bbegin, bend = int(begin/block_sz), int(end/block_sz)
        block_indices = [i for i in range(len(seq[:end])//block_sz)]
        random.shuffle(block_indices)
        for i in block_indices[bbegin:bend]:
            sseq += copied_seq[i*block_sz:(i+1)*block_sz]
    return sseq    
    

    
def plotP0(seq, blocks, snbr):
    tree_l = max(seq)+1
    plt.figure(1)
    hlocs_stat = np.zeros(maxh-1)
    for i in range(tree_l):
        locs = np.array([j for j in range(len(seq)) if seq[j]==i])
        locss = deepcopy(locs)
        locss[:-1] = locss[1:]
        locsd = locss-locs
        bins = range(maxh)
        hlocs = np.histogram(locsd, bins, density=True)
        hlocs_stat = hlocs_stat + hlocs[0]/tree_l
    plt.loglog(bins[:-1], hlocs_stat, label='original') 
    for nfig, block in enumerate(blocks):
        hlocs_stat = np.zeros(maxh-1)
        shuffleseq = shuffleblocks(seq, block, snbr)
        plt.figure(nfig+2)
        plt.plot(shuffleseq)
        plt.title(block)
        for i in range(tree_l):
            locs = np.array([j for j in range(len(shuffleseq)) if shuffleseq[j]==i])
            locss = deepcopy(locs)
            locss[:-1] = locss[1:]
            locsd = locss-locs
            bins = range(maxh)
            hlocs = np.histogram(locsd, bins, density=True)
            hlocs_stat = hlocs_stat + hlocs[0]/tree_l
            
        plt.figure(1)    
        plt.loglog(bins[:-1], hlocs_stat, label=block, alpha=0.5) 
        plt.legend()
   
   
    

plt.figure(10)

tree_depth = 6
T = 0.6

#tree_depth = 7
#T = 1.1
testseq = sequence_generator.um_sequence_generator(0, 100000, 1, T, tree_depth, 2)[0]
plt.plot(testseq)
plotP0(testseq[:100000], blocks, 10)
plt.title('tree depth %i Temperature %.2f' % (tree_depth, T))
plt.xlim(left=1)

#
#
#plt.plot(original[:100000])
#plotP0(original[:100000], blocks, 10)
#plt.xlim(left=1)

    
    
#shuffledtest = shuffleblocks(original[:1000000], 10000, 100)    
#plt.figure()
#plt.plot(original[:1000000])
#plt.figure()
#plt.plot(shuffledtest)