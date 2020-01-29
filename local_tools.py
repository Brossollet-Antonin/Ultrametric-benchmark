# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 17:39:14 2020

@author: Simon
"""

import numpy as np
from copy import deepcopy
import random

def verbose(message, args, lvl=1):
	if args.verbose >= lvl:
		print(message)

def base_conv(value, base):
    # Equivalent to bin(value) but for an arbitrary base. Return a string in the given base
    res = ''
    while value > 0:
        res = str(value % base) + res 
        value = value//base
    return res

def make_ohe(y, n_labels):
    ohe = np.zeros((len(y), n_labels))    
    ohe[np.arange(len(y)),y] = 1
    return ohe

######################################

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