# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 11:08:23 2019

@author: Antonin
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
import time

parser = argparse.ArgumentParser('./main.py', description='Run test')
parser.add_argument('--gpu', action='store_true', dest='cuda', help="Use GPU")
parser.add_argument('--savefolder', type=str, default='./', help="Folder to save the data")

# dataset parameters
data_params = parser.add_argument_group('Dataset Parameters')
data_params.add_argument('--dataset', type=str, dest='data_origin', default='CIFAR100', choices=['MNIST', 'CIFAR10', 'CIFAR100'])

# model/hyperparameters parameters
model_params = parser.add_argument_group('Model Parameters')
model_params.add_argument('--lr', type=int, default=0.01, help='Learning rate')
model_params.add_argument('--minibatch', type=int, dest='minibatches_list', nargs='*', default=[10], help='Size of the training mini-batches')
model_params.add_argument('--memory', type=int, dest='memory_list', nargs='*', default=[0], help='Size of the memory for replay training')
model_params.add_argument('--nbrtest', type=int, default=10, dest='test_nbr', help='Number of data points to get during training (number of test of the dataset')

# sequence parameters
seq_params = parser.add_argument_group('Sequence Parameters')
seq_params.add_argument('--seqlength', type=int, default=100000, dest='sequence_length', help='Length of the training sequence')
seq_params.add_argument('--blocksz', type=int, dest='block_size_shuffle_list', nargs='*', default=[100], help='Size of the block used to shuffle the sequence')
seq_params.add_argument('-T', '--temperature', type=float, dest='temperature_list', nargs='*', default=[0.6], help='Temperature for the random walk (the energy step is by default equal to 1)')

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

    if args.data_origin=='MNIST' or args.data_origin=='CIFAR10':
        depth, tree_branching= 3, 2
    elif args.data_origin=='CIFAR100':
        depth, tree_branching= 6, 2

    dataset = artificial_dataset.artificial_dataset(data_origin=args.data_origin)
    data_branching = tree_branching

    for minibatches in args.minibatches_list:
        for memory_sz in args.memory_list:
            for block_size_shuffle in args.block_size_shuffle_list:
                    for T in args.temperature_list:
                        savepath = "./Results/%s/%s/length%d_batches%d/" % (args.savefolder, args.data_origin, args.sequence_length, minibatches)
                        save_folder = "T%.3f Memory%d block%d %.3f" % (T, memory_sz, block_size_shuffle, systime)
                        os.makedirs(savepath + save_folder)

                        parameters = np.array([[T, depth, tree_branching, args.sequence_length, minibatches, block_size_shuffle, args.test_nbr, step, memory_sz,
                                                args.lr, args.data_origin, systime, 'GPU' if args.cuda else 'CPU', args.nnarchi],
                                               ["Temperature", "Tree Depth", "Tree Branching", "Sequence Length", "Minibatches Size",
                                                "Size Blocks Shuffle", "Number of tests", "Energy Step", "Replay Memory Size",
                                                "Learning rate", "Dataset", "Random Seed", "CPU/GPU?", "NN architecture"]])


                        num_classes = 100 if args.data_origin=='CIFAR100' else 10
                        netfc_original = neuralnet.Net_CNN(dataset.data_origin) if args.nnarchi=='CNN' else neuralnet.resnetN(type=args.resnettype, num_classes=num_classes, data_origin=dataset.data_origin)
                        netfc_original.to(device)

                        netfc_shuffle = neuralnet.Net_CNN(dataset.data_origin) if args.nnarchi=='CNN' else neuralnet.resnetN(type=args.resnettype, num_classes=num_classes, data_origin=dataset.data_origin)
                        netfc_shuffle.to(device)

                        trainer = algo.training('temporal correlation', 'reservoir sampling', dataset=dataset,
                        task_sz_nbr=minibatches,
                        tree_depth=depth, preprocessing=False, device=device, sequence_length=args.sequence_length, energy_step=step, T=T,
                        tree_branching=tree_branching)

                        exec(open("/rigel/theory/users/ab4877/Ultrametric-benchmark/testermain.py").read())

                        diagnos_original = diagnosis.hierarchical_error(netfc_original, trainer, device)
                        diagnos_shuffle = diagnosis.hierarchical_error(netfc_shuffle, trainer, device)

                        exec(open("/rigel/theory/users/ab4877/Ultrametric-benchmark/save_data.py").read())



if __name__ == '__main__':
    args = parser.parse_args()
    run(args)
