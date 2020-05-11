# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 15:02:44 2020

@author: Simon Lebastard
"""

import argparse
import os, sys, pdb
import getpass

import random
from datetime import datetime
import time

import numpy as np
import torch

from local_tools import verbose

import dataset as ds
import neuralnet

import utils
from trainer import Trainer
from ultrametric_analysis import ultrametric_analysis
from data_saver import save_results

paths = utils.get_project_paths()
cwd = paths['root']

parser = argparse.ArgumentParser('./main.py', description='Run test')
parser.add_argument('--gpu', action='store_true', dest='cuda', help="Use GPU")
parser.add_argument('--savefolder', type=str, default='./', help="Folder to save the data")
parser.add_argument('-v', '--verbose', dest='verbose', action='count', default=0)

# dataset parameters
data_params = parser.add_argument_group('Dataset Parameters')
data_params.add_argument('--dataset', type=str, dest='data_origin', default='CIFAR100', choices=['MNIST', 'CIFAR10', 'CIFAR100', 'artificial'])
data_params.add_argument('--data_tree_depth', type=int, dest='artif_tree_depth', default=3)
data_params.add_argument('--data_flips_rate', type=float, default=0.04)
data_params.add_argument('--data_seq_size', type=int, dest='artif_seq_size', default=200)
data_params.add_argument('--shuffle_classes', type=int, dest='artif_shuffle_classes', default=1)
data_params.add_argument('--proba_transition', type=float, default=0.1)
data_params.add_argument('--split_length', type=int, dest='split_length_list', nargs='*', default=[100])

# model/hyperparameters parameters
model_params = parser.add_argument_group('Model Parameters')
model_params.add_argument('--lr', type=int, default=0.01, help='Learning rate')
model_params.add_argument('--minibatch', type=int, dest='minibatches_list', nargs='*', default=[10], help='Size of the training mini-batches')
model_params.add_argument('--memory', type=int, dest='memory_list', nargs='*', default=[0], help='Size of the memory for replay training')
model_params.add_argument('--nbrtest', type=int, default=100, dest='test_nbr', help='Number of data points to get during training (number of test of the dataset')

# sequence parameters
seq_params = parser.add_argument_group('Sequence Parameters')
seq_params.add_argument('--seqtype', type=str, default='ultrametric', dest='sequence_type', choices=['ultrametric', 'spatial_correlation', 'random', 'uniform', 'ladder_blocks1', 'random_blocks1', 'ladder_blocks2', 'random_blocks2', 'random_blocks2_2freq'], help='Method used to generate the training sequence')
seq_params.add_argument('--seqlength', type=int, default=100000, dest='sequence_length', help='Length of the training sequence')
seq_params.add_argument('--blocksz', type=int, dest='block_size_shuffle_list', nargs='*', default=[], help='Size of the block used to shuffle the sequence')
seq_params.add_argument('-T', '--temperature', type=float, dest='temperature_list', nargs='*', default=[0.4], help='Temperature for the random walk (the energy step is by default equal to 1)')
seq_params.add_argument('--force_switch', type=int, default=1, help='When true, the training sequence cannot remain at the same value from one state to the next through time')
seq_params.add_argument('--min_state_visit', type=int, default=0, help='Indicated the number of times each state must be visited in the generated training sequence (no constraint by default)')
seq_params.add_argument('--T_adaptive', type=float, default=0, help='When specified, a temperature is computed so that a all states have an inbound transition probability above the given threshold')

# neural network parameters
nn_params = parser.add_argument_group('Neural Network Parameters')
nn_params.add_argument('--nnarchi', type=str, default='ResNet', choices=['FCL', 'CNN', 'ResNet'], help='Architure of the neural network used')
nn_params.add_argument('--resnettype', type=int, default=50, choices=[18, 34, 50, 101, 152], help='Type of ResNet network to use')
nn_params.add_argument('--hidden_sizes', type=int, default=20, help='A list of hidden sizes in case of a FCL network')

def run(args):
	step = 1
	args.test_stride = int(args.sequence_length/args.test_nbr)
	systime = time.time()
	random.seed(systime)

	args.enable_shuffling = True
	if not args.block_size_shuffle_list:
		args.enable_shuffling = False
	if args.sequence_type == 'uniform':
		args.enable_shuffling = False

	device = torch.device('cuda') if args.cuda else torch.device('cpu')

	verbose('Generating dataset {0:s} - data_seq_size={1:d}'.format(args.data_origin, args.artif_seq_size), args)

	dataset = ds.Dataset(
		data_origin = args.data_origin,
		data_sz=args.artif_seq_size,
		tree_depth = args.artif_tree_depth,
		class_sz_train=1000,
		class_sz_test=400,
		ratio_type='linear',
		ratio_value=args.data_flips_rate,
		noise_level=3,
		shuffle_classes=args.artif_shuffle_classes
		)

	verbose('Done generating dataset {0:s}'.format(args.data_origin), args)

	for batch_sz in args.minibatches_list:
		for memory_sz in args.memory_list:
			for T in args.temperature_list:
				save_root = paths['simus'] + "1toM/%s_%s/%s/%s_length%d_batches%d/" % (args.data_origin, dataset.num_classes, args.nnarchi, args.sequence_type, args.sequence_length, batch_sz)
				if dataset.data_origin == 'artificial':
					if args.nnarchi == 'FCL':
						save_root = paths['simus'] + "1toM/%s_%s/%s%d/%s_length%d_batches%d_seqlen%d_ratio%d" % (args.data_origin, dataset.num_classes, args.nnarchi, args.hidden_sizes, args.sequence_type, args.sequence_length, batch_sz, args.artif_seq_size, dataset.ratio_value)							
					else:
						save_root = paths['simus'] + "1toM/%s_%s/%s/%s_length%d_batches%d_seqlen%d_ratio%d" % (args.data_origin, dataset.num_classes, args.nnarchi, args.sequence_type, args.sequence_length, batch_sz, args.artif_seq_size, dataset.ratio_value)
				if 'blocks' in args.sequence_type:
					T = float(0)
					if args.sequence_type == 'random_blocks2_2freq':
						save_root = save_root+"_splitlengths"+str(args.split_length_list[0])+"_"+str(args.split_length_list[1])+"/"
					else:
						save_root = save_root+"_splitlength"+str(args.split_length_list[0])+"/"
				else:
					save_root = save_root+"/"

				if (args.artif_shuffle_classes==0):
					save_root = save_root[:-1]+"_noclassreshuffle/"

				#save_folder = "T%.3f_Memory%d_block%d_%s" % (T, memory_sz, block_size_shuffle, datetime.now().strftime("%y%m%d_%H%M%S"))

				if args.sequence_type == 'uniform':
					T = 0.0

				parameters = {
					"Temperature": T,
					"Tree Depth": dataset.depth,
					"Tree Branching": dataset.branching,
					"Flips ratio": args.data_flips_rate,
					"Sequence Length": args.sequence_length,
					"Minibatches Size": batch_sz,
					"Number of tests": args.test_nbr,
					"Energy Step": step,
					"Replay Memory Size": memory_sz,
					"Learning rate": args.lr,
					"Dataset": args.data_origin,
					"Random Seed": systime,
					"device_type": 'GPU' if args.cuda else 'CPU',
					"NN architecture": args.nnarchi,
					"Split total length": args.split_length_list[0],
					"Original command": str(sys.argv) # We store the original command for this set of simulations
				}
				# ToDo: - turn parameters into a dictionnary
				#       - export as JSON

				verbose(
					'Instanciating network and trainer (sequence generation with {0:s}, length {1:d})...'.format(args.sequence_type, args.sequence_length),
					args
					)

				if args.nnarchi == 'FCL':
					netfc_original = neuralnet.Net_FCL(dataset, args.hidden_sizes)
				elif args.nnarchi == 'CNN':
					netfc_original = neuralnet.Net_CNN(dataset)
				else:
					netfc_original = neuralnet.resnetN(type=args.resnettype, dataset=dataset)
				netfc_original.to(device)

				if args.nnarchi == 'FCL':
					netfc_shuffle = neuralnet.Net_FCL(dataset, args.hidden_sizes)
				elif args.nnarchi == 'CNN':
					netfc_shuffle = neuralnet.Net_CNN(dataset)
				else:
					netfc_shuffle = neuralnet.resnetN(type=args.resnettype, dataset=dataset)
				netfc_shuffle.to(device)

				trainer = Trainer(
					dataset = dataset,
					network = netfc_original,
					training_type = args.sequence_type,
					memory_sampling = 'reservoir sampling',
					memory_sz = memory_sz,
					batch_sz = batch_sz,
					preprocessing = False,
					device = device,
					min_visit = args.min_state_visit,
					sequence_length = args.sequence_length,
					energy_step = step,
					proba_transition = args.proba_transition,
					T = T,
					dynamic_T_thr = args.T_adaptive,
					split_length_list = args.split_length_list
					)
				trainer.network_orig = netfc_original
				trainer.network_shfl = netfc_shuffle

				verbose('...done', args)

				rs = ultrametric_analysis(trainer, args, args.block_size_shuffle_list)
				rs.parameters = parameters
				rs.T = trainer.T
				rs.memory_sz = memory_sz

				save_results(rs, save_root)



if __name__ == '__main__':
	args = parser.parse_args()
	run(args)
