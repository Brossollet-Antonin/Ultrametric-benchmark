import os
import numpy as np
from scipy.io import savemat
import pdb

# rootdir = '/home/proprietaire/Documents/Workspace/Jobs/Columbia/ultrametric_benchmark/Ultrametric-benchmark/Results/1toM/artificial_32/FCL20/ultrametric_length300000_batches10_seqlen200_ratio8'
# rootdir = '/home/proprietaire/Documents/Workspace/Jobs/Columbia/ultrametric_benchmark/Ultrametric-benchmark/Results/1toM/artificial_32/FCL20/random_blocks2_length300000_batches10_seqlen200_ratio8_splitlength1000'
rootdir = '/home/proprietaire/Documents/Workspace/Jobs/Columbia/ultrametric_benchmark/Ultrametric-benchmark/Results/1toM/artificial_32/FCL20/uniform_length300000_batches10_seqlen200_ratio8'

pdb.set_trace()
simus_root = os.listdir(rootdir)

for simu_subdir in simus_root:
    os.chdir(rootdir+'/'+simu_subdir)
    orig_acc_npy = np.load('var_original_accuracy.npy')
    savemat(
        'var_orig_acc.mat',
        { 'accuracy': orig_acc_npy }
    )
    shuffle_roots = os.listdir(rootdir+'/'+simu_subdir)
    for shuffle_subdir in shuffle_roots:
        if 'shuffle_' not in shuffle_subdir:
            continue
        os.chdir(rootdir+'/'+simu_subdir+'/'+shuffle_subdir)
        shfl_size = shuffle_subdir.split('_')[1]
        shfl_acc_npy = np.load('var_shuffle_accuracy.npy')
        os.chdir(rootdir+'/'+simu_subdir)
        savemat(
            'var_shfl_'+str(shfl_size)+'_acc.mat',
            { 'accuracy': shfl_acc_npy }
        )
