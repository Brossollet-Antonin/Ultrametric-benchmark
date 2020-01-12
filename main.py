# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 15:02:44 2020

@author: Simon Lebastard
"""

import argparse
import os
import diagnosis
import numpy as np
import artificial_dataset
import random
import torch 
import neuralnet
import algo
from datetime import datetime
import time

cwd = os.getcwd()

parser = argparse.ArgumentParser('./main.py', description='Run test')
parser.add_argument('--gpu', action='store_true', dest='cuda', help="Use GPU")
parser.add_argument('--savefolder', type=str, default='./', help="Folder to save the data")
parser.add_argument('-v', '--verbose', dest='verbose', action='count', default=0)

# dataset parameters
data_params = parser.add_argument_group('Dataset Parameters')
data_params.add_argument('--dataset', type=str, dest='data_origin', default='CIFAR100', choices=['MNIST', 'CIFAR10', 'CIFAR100', 'artificial_8', 'artificial_16', 'artificial_32'])
data_params.add_argument('--data_seq_size', type=int, dest='artif_seq_size', default=200)

# model/hyperparameters parameters
model_params = parser.add_argument_group('Model Parameters')
model_params.add_argument('--lr', type=int, default=0.01, help='Learning rate')
model_params.add_argument('--minibatch', type=int, dest='minibatches_list', nargs='*', default=[10], help='Size of the training mini-batches')
model_params.add_argument('--memory', type=int, dest='memory_list', nargs='*', default=[0], help='Size of the memory for replay training')
model_params.add_argument('--nbrtest', type=int, default=100, dest='test_nbr', help='Number of data points to get during training (number of test of the dataset')

# sequence parameters
seq_params = parser.add_argument_group('Sequence Parameters')
seq_params.add_argument('--seqtype', type=str, default='temporal_correlation', dest='sequence_type', choices=['temporal_correlation', 'spatial_correlation', 'random', 'uniform', 'onefold_split', 'twofold_split'], help='Method used to generate the training sequence')
seq_params.add_argument('--seqlength', type=int, default=100000, dest='sequence_length', help='Length of the training sequence')
seq_params.add_argument('--blocksz', type=int, dest='block_size_shuffle_list', nargs='*', default=[100], help='Size of the block used to shuffle the sequence')
seq_params.add_argument('-T', '--temperature', type=float, dest='temperature_list', nargs='*', default=[0.4], help='Temperature for the random walk (the energy step is by default equal to 1)')

# neural network parameters
nn_params = parser.add_argument_group('Neural Network Parameters')
nn_params.add_argument('--nnarchi', type=str, default='ResNet', choices=['CNN', 'ResNet'], help='Architure of the neural network used')
nn_params.add_argument('--resnettype', type=int, default=50, choices=[18, 34, 50, 101, 152], help='Type of ResNet network to use')


def run(args):
    step = 1
    test_stride = int(args.sequence_length/args.test_nbr)
    systime = time.time()
    random.seed(systime)
    
    device = torch.device('cuda') if args.cuda else torch.device('cpu')
        
    if args.verbose:
        print('Generating dataset {0:s} - data_seq_size={1:d}'.format(args.data_origin, args.artif_seq_size))

    dataset = artificial_dataset.artificial_dataset(
        data_origin = args.data_origin,
        data_sz=args.artif_seq_size,
        class_sz_train=1000,
        class_sz_test=400,
        ratio_type='linear',
        ratio_value=1,
        noise_level=1
        )

    if args.verbose:
        print('Done generating dataset {0:s}'.format(args.data_origin))
    
    for minibatches in args.minibatches_list:
        for memory_sz in args.memory_list:
            for block_size_shuffle in args.block_size_shuffle_list:
                for T in args.temperature_list:                         
                    savepath = cwd+"/Results/%s/%s/length%d_batches%d/" % (args.data_origin, args.nnarchi, args.sequence_length, minibatches)
                    #save_folder = "T%.3f_Memory%d_block%d_%.3f" % (T, memory_sz, block_size_shuffle, systime)
                    save_folder = "T%.3f_Memory%d_block%d_%s" % (T, memory_sz, block_size_shuffle, datetime.now().strftime("%y%m%d_%H%M%s"))
            
                    parameters = np.array([[T, dataset.depth, dataset.branching, args.sequence_length, minibatches, block_size_shuffle, args.test_nbr, step, memory_sz, 
                                            args.lr, args.data_origin, systime, 'GPU' if args.cuda else 'CPU', args.nnarchi],
                                           ["Temperature", "Tree Depth", "Tree Branching", "Sequence Length", "Minibatches Size", 
                                            "Size Blocks Shuffle", "Number of tests", "Energy Step", "Replay Memory Size", 
                                            "Learning rate", "Dataset", "Random Seed", "CPU/GPU?", "NN architecture"]])
                    # ToDo: - turn parameters into a dictionnary
                    #       - export as JSON

                    if args.verbose:
                        print('Instanciating network and trainer (sequence generation with {0:s}, length {1:d})...'.format(args.sequence_type, args.sequence_length))

                    netfc_original = neuralnet.Net_CNN(dataset) if args.nnarchi=='CNN' else neuralnet.resnetN(type=args.resnettype, dataset=dataset)
                    netfc_original.to(device)
                    
                    netfc_shuffle = neuralnet.Net_CNN(dataset) if args.nnarchi=='CNN' else neuralnet.resnetN(type=args.resnettype, dataset=dataset)
                    netfc_shuffle.to(device)
                    
                    args.sequence_type = args.sequence_type.replace('_', ' ')
                    trainer = algo.training(
                        args.sequence_type, 'reservoir sampling',
                        dataset=dataset,
                        task_sz_nbr=minibatches,
                        preprocessing=False,
                        device=device,
                        sequence_length=args.sequence_length,
                        energy_step=step,
                        T=T
                        )
                    
                    if args.verbose:
                        print('...done')

                    exec(
                        open("./testermain.py", encoding="utf-8").read()
                        ) 
                                
                    diagnos_original = diagnosis.hierarchical_error(netfc_original, trainer, device)
                    diagnos_shuffle = diagnosis.hierarchical_error(netfc_shuffle, trainer, device)
        
                    exec(
                        open("./save_data.py", encoding="utf-8").read()
                        )
            
    

if __name__ == '__main__':
    args = parser.parse_args()
    run(args)
