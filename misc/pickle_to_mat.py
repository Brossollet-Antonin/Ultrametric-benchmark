import os
import numpy as np
from scipy.io import savemat
import pdb

rootdir = '/home/slebastard-adc/Documents/Projects/ultrametric_benchmark/Ultrametric-Benchmark/Results/1toM/artificial_128/FCL50/'
#um_root = '/home/proprietaire/Documents/Workspace/Jobs/Columbia/ultrametric_benchmark/Ultrametric-benchmark/Results/1toM/artificial_32/FCL20/ultrametric_length300000_batches10_seqlen200_ratio8_noclassreshuffle'
um_root = rootdir+'ultrametric_length300000_batches10_seqlen200_ratio8'
rb_root = rootdir+'random_blocks2_length300000_batches10_seqlen200_ratio8_splitlength328'
uni_root = rootdir+'uniform_length200000_batches10_seqlen200_ratio8'

simus_root = os.listdir(um_root)

for simu_subdir in simus_root:
    os.chdir(um_root+'/'+simu_subdir)
    orig_seq_pickle = np.load('train_labels_orig.pickle', allow_pickle=True)
    savemat(
        'train_labels_orig.mat',
        { 'sequence': orig_seq_pickle }
    )
    shuffle_roots = os.listdir(um_root+'/'+simu_subdir)
    for shuffle_subdir in shuffle_roots:
        if 'shuffle_' not in shuffle_subdir:
            continue
        os.chdir(um_root+'/'+simu_subdir+'/'+shuffle_subdir)
        shfl_size = shuffle_subdir.split('_')[1]
        shfl_seq_pickle = np.load('train_labels_shfl.pickle', allow_pickle=True)
        os.chdir(um_root+'/'+simu_subdir)
        savemat(
            'train_labels_shfl_'+str(shfl_size)+'.mat',
            { 'sequence': shfl_seq_pickle }
        )

simus_root = os.listdir(rb_root)

for simu_subdir in simus_root:
    os.chdir(rb_root+'/'+simu_subdir)
    orig_seq_pickle = np.load('train_labels_orig.pickle', allow_pickle=True)
    savemat(
        'train_labels_orig.mat',
        { 'sequence': orig_seq_pickle }
    )
    shuffle_roots = os.listdir(rb_root+'/'+simu_subdir)
    for shuffle_subdir in shuffle_roots:
        if 'shuffle_' not in shuffle_subdir:
            continue
        os.chdir(rb_root+'/'+simu_subdir+'/'+shuffle_subdir)
        shfl_size = shuffle_subdir.split('_')[1]
        shfl_seq_pickle = np.load('train_labels_shfl.pickle', allow_pickle=True)
        os.chdir(rb_root+'/'+simu_subdir)
        savemat(
            'train_labels_shfl_'+str(shfl_size)+'.mat',
            { 'sequence': shfl_seq_pickle }
        )

simus_root = os.listdir(uni_root)

for simu_subdir in simus_root:
    os.chdir(uni_root+'/'+simu_subdir)
    orig_seq_pickle = np.load('train_labels_orig.pickle', allow_pickle=True)
    savemat(
        'train_labels_orig.mat',
        { 'sequence': orig_seq_pickle }
    )

