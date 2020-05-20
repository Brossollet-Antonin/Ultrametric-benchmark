# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 17:39:14 2020

@author: Simon
"""

import numpy as np
from copy import deepcopy
import random
import os
import pickle
import scipy.io
import pdb

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

######################################

def simulate_sequence(T_list, seq_length=200000, tree_depth=3, tree_branching=2, minimum_classcount=0, rate_law='power', force_switch=True):
    from sequence_generator_temporal import TempCorr_SequenceGenerator
    from ultrametric_analysis import ResultSet
    from matplotlib import pyplot as plt

    n_Ts = len(T_list)
    assert (n_Ts>0)
    lbls_fig = plt.figure(figsize=(18,10*n_Ts))

    seqgen = TempCorr_SequenceGenerator()
    for T_id, T in enumerate(T_list):
        sequence_labels, rates = seqgen.generate_labels(
            sequence_first=0,
            sequence_length=seq_length,
            energy_step=1,
            T=T,
            tree_depth=tree_depth,
            tree_branching=tree_branching,
            minimum_classcount=minimum_classcount,
            rate_law=rate_law,
            force_switch=force_switch,
            dynamic_T=0
        )
        lbls_ax = plt.subplot(n_Ts, 1, 1+T_id)
        lbls_ax.plot(sequence_labels)
        plt.ylim((0, tree_branching**tree_depth))
        ttl = 'History of labels in the original training sequence - T='+str(T)
        plt.title(ttl)

def get_lbl_distr(shuffled_sequence, min_range, max_range, n_classes):
    """
    Inputs:
    shuffled_sequence: list
    min_range: int
    max_range: int

    Returns:
    histogram of the labels distribution from min_range to max_range
    """
    hist_tuple = np.histogram(
        shuffled_sequence[min_range:max_range],
        bins = n_classes
        )

    return hist_tuple[0]

#######################################

def generate_mat_sequences(rootdir, outroot, block_sizes):
    for subdir_fp, dirs, files in os.walk(rootdir):
        if (subdir_fp == rootdir):
            continue
        os.chdir(subdir_fp)
        subdir = subdir_fp.split('/')[-1]

        # Translate the original sequence
        if 'train_labels_orig.pickle' in files:
            lbl_seq = pickle.load(open('train_labels_orig.pickle', "rb"))
            scipy.io.savemat(outroot+'/lblseq_orig_'+subdir, mdict={'lbl_seq': lbl_seq})

        # Translate all shuffled sequences found
        for block_sz in block_sizes:
            shfl_subdir = 'shuffle_'+str(block_sz)
            if shfl_subdir in dirs:
                lbl_seq = pickle.load(open(shfl_subdir+'/train_labels_shfl.pickle', "rb"))
                scipy.io.savemat(outroot+'/lblseq_shfl'+str(block_sz)+'_'+subdir, mdict={'lbl_seq': lbl_seq})