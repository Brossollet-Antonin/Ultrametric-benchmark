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
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from utils import verbose

import dataset as ds
import neuralnet

import utils
from trainer import Trainer
from ultrametric_analysis import train_sequenceset
import data_saver

class ResultSet:
    def __init__(self):
        pass

class OrigCP:
	def __init__(self, orig_path):

		# Assert that the checkpoint path actually exist and refers to a directory
		assert os.path.isdir(os.path.join(paths['simus'], orig_path)), "Provided orig_path is not a valid directory, yields: {}".format(os.path.join(paths['simus'], orig_path))
		orig_folders = orig_path.split("/")
		self.root = os.path.join(
			paths['simus'],
			"/".join(orig_folders[:-1])
		)
		self.subfolder = orig_folders[-1]

		# Load the original sequence
		import pickle
		with open(os.path.join(self.root, self.subfolder, 'train_labels_orig.pickle'), 'rb') as file:
			self.train_sequence = pickle.load(file)

		# Load the parameters file
		import json
		with open(os.path.join(self.root, self.subfolder, 'parameters.json'), 'r') as param_file:
			self.parameters = json.load(param_file)


paths = utils.get_project_paths()

parser = argparse.ArgumentParser(os.path.join(paths['root'], "main.py"), description='Run test')
parser.add_argument('--gpu', action='store_true', dest='cuda', help="Use GPU")
parser.add_argument('--savefolder', type=str, default='./', help="Folder to save the data")
parser.add_argument('--verbose', type=int, dest='verbose', default=0)
parser.add_argument('--use_orig', type=str, dest="orig_path", default="", help="If you want the simulations to pick up from a generated original sequence, provide the path to the corresponding subfolder, starting from ./Results/ (Ex:'1toM/MNIST_8/CNN256/ultrametric_length4000000_nosplit/T0.225_200903_164112'). Only the shuffle will be executed.")

# dataset and ultrametric tree parameters
data_params = parser.add_argument_group('Dataset Parameters')

data_params.add_argument('--dataset', type=str, dest='data_origin', default='CIFAR100', choices=['MNIST', 'CIFAR10', 'CIFAR100', 'artificial'])

data_params.add_argument('--data_tree_depth', type=int, dest='artif_tree_depth', default=3)
data_params.add_argument('--data_flips_rate', type=float, default=0.04)
data_params.add_argument('-T', '--temperature', type=float, dest='T', default=0.4, help='Temperature for the random walk (the energy step is by default equal to 1)')
data_params.add_argument('--shuffle_classes', type=int, dest='artif_shuffle_classes', default=1)
data_params.add_argument('--proba_transition', type=float, default=0.1)
data_params.add_argument('--data_seq_size', type=int, dest='artif_seq_size', default=200)

# model/hyperparameters parameters
model_params = parser.add_argument_group('Model Parameters')
model_params.add_argument('--nnarchi', type=str, default='FCL', choices=['FCL', 'CNN', 'ResNet'], help='Architure of the neural network used')
model_params.add_argument('--resnettype', type=int, default=50, choices=[18, 34, 50, 101, 152], help='Type of ResNet network to use')
model_params.add_argument('--hidden_sizes', type=int, nargs='*', default=[20], help='Provides the number of filters per convolutional layer (and thus, the number of conv layers) in the case of a CNN, and the number of hidden units per hidden layer in the case of a FCL network')
model_params.add_argument('--nonlin', type=str, default='none', choices=['none', 'relu'], help='Only in the case of a MLP, provides the type of nonlinearity (or the absence of it) between each layer')
model_params.add_argument('--lr', type=float, default=0.01, help='Learning rate')
model_params.add_argument('--batch_sz', type=int, default=10, help='Size of the training mini-batches')
model_params.add_argument('--memory_sz', type=int, default=0, help='Size of the memory for replay training')
model_params.add_argument('--loss_fn', type=str, default='cross_entropy', help='Loss function')
model_params.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adam', 'adagrad'], help='Optimizer used for training')

# "Memory allocation" parameters, from methods that aim specifically at mitigating catastrophic interference
cl_params = parser.add_argument_group('Continual leanrning capabilities')
cl_params.add_argument('--ewc', action='store_true', help="use 'EWC' (Kirkpatrick et al, 2017)")
cl_params.add_argument('--lambda', type=float, dest="ewc_lambda", default=0.0, help="--> EWC: regularisation strength")
cl_params.add_argument('--fisher_n', type=int, help="--> EWC: sample size estimating Fisher Information")
cl_params.add_argument('--gamma', type=float, default=0.95, help="--> EWC: forgetting coefficient (for 'online EWC')")
cl_params.add_argument('--emp-fi', action='store_true', help="--> EWC: estimate FI with provided labels")

# sequence parameters
seq_params = parser.add_argument_group('Sequence Parameters')
seq_params.add_argument('--seqtype', type=str, default='ultrametric', dest='sequence_type', choices=['ultrametric', 'uniform', 'ladder_blocks1', 'random_blocks1', 'ladder_blocks2', 'random_blocks2', 'random_blocks2_2freq'], help='Method used to generate the training sequence')
seq_params.add_argument('--seqlength', type=int, default=300000, dest='sequence_length', help='Length of the training sequence')
seq_params.add_argument('--nbrtest', type=int, default=300, dest='test_nbr', help='Number of data points to get during training (number of test of the dataset')
seq_params.add_argument('--blocksz', type=int, dest='block_size_shuffle_list', nargs='*', default=[], help='Size of the block used to shuffle the sequence')
seq_params.add_argument('--force_switch', type=int, default=1, help='When true, the training sequence cannot remain at the same value from one state to the next through time')
seq_params.add_argument('--min_state_visit', type=int, default=0, help='Indicated the number of times each state must be visited in the generated training sequence (no constraint by default)')
seq_params.add_argument('--T_adaptive', type=float, default=0, help='When specified, a temperature is computed so that a all states have an inbound transition probability above the given threshold')
seq_params.add_argument('--split_length', type=int, dest='split_length_list', nargs='*', default=[100], help='Size of task sequence in the RB scenario')
seq_params.add_argument('--save_um_distances', action='store_true', default=False, help='If specified, will store ultrametric distances between predictions and ground truth on test set at each evaluation point in the sequence')

def run(args):
	# Global parameters
	step = 1 # Energy step
	args.test_stride = int(args.sequence_length/args.test_nbr) # Number of sequence samples the model learns on between two evaluation steps
	systime = time.time()
	random.seed(systime)

	# This will control whether we run any shuffling scenario or not (those are demanding in computational resources)
	if ((not args.block_size_shuffle_list) or args.block_size_shuffle_list==[0] or args.sequence_type == 'uniform'):
		args.enable_shuffling = False
	else:
		args.enable_shuffling = True
	# 0 is passed as a dummy block size by our slurm batch script, it should be removed
	if 0 in args.block_size_shuffle_list:
		args.block_size_shuffle_list.remove(0)

	device = torch.device('cuda') if args.cuda else torch.device('cpu')

	#------------------------------#
	#----- DATASET GENERATION -----#
	#------------------------------#

	verbose('Generating dataset {0:s} - data_seq_size={1:d}'.format(args.data_origin, args.artif_seq_size), args.verbose, 0)

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

	verbose('Done generating dataset {0:s}'.format(args.data_origin), args.verbose, 0)

	#----------------------------------#
	#----- PREPARING FILE OUTPUTS -----#
	#----------------------------------#

	cl_strategy = 'EWC' if args.ewc else '1toM'

	if args.orig_path != "":
		verbose("Attempting to run simulations from checkpoint", args.verbose, 0)
		orig_checkpoint = OrigCP(args.orig_path)
		save_root = orig_checkpoint.root
		if 'blocks' in args.sequence_type:
			args.T = float(0)
		verbose("Save root set to {:s}".format(orig_checkpoint.root), 0)

	else:
		orig_checkpoint = None
		verbose("Running simulations from scratch (default, no checkpoint used)", args.verbose, 0)
		save_root = os.path.join(
			paths['simus'],
			"{cl_strat:s}/{data_origin:s}_{n_classes:d}/{nnarchi:s}{hidlay_width:s}/{seq_type:s}_length{seq_length:d}_batches{batch_size:d}_optim{optimizer:s}".format(
				cl_strat = cl_strategy,
				data_origin = args.data_origin,
				n_classes = dataset.num_classes,
				nnarchi = args.nnarchi,
				hidlay_width = "x".join([str(el) for el in args.hidden_sizes]),
				seq_type = args.sequence_type,
				seq_length = args.sequence_length,
				batch_size = args.batch_sz,
				optimizer = args.optimizer
			)
		)
		if args.nonlin == 'relu':
			save_root += "_nonlinRelu"
		if dataset.data_origin == 'artificial':
			save_root += "_seqlen{patterns_size:d}_ratio{bitflips:d}" .format(
				patterns_size = args.artif_seq_size,
				bitflips = int(dataset.data_sz*dataset.ratio_value)
			)
		if 'blocks' in args.sequence_type:
			args.T = float(0)
			if args.sequence_type == 'random_blocks2_2freq':
				save_root += "_splitlengths"+str(args.split_length_list[0])+"_"+str(args.split_length_list[1])
			else:
				save_root += "_splitlength"+str(args.split_length_list[0])

		if (args.artif_shuffle_classes==0):
			save_root += "_noclassreshuffle"

	verbose("Output directory for this simulation set: {:s}".format(save_root), args.verbose, 0)

	if args.sequence_type == 'uniform':
		args.T = 0.0


	verbose(
		'Instanciating network and trainer (sequence generation with {0:s}, length {1:d})...'.format(args.sequence_type, args.sequence_length),
		args.verbose, 0
		)

	#------------------------------#
	#----- MODEL (CLASSIFIER) -----#
	#------------------------------#
	# neuralnet models are now subclasses of ContinualLearner and can all implement CL strategies such as EWC

	if args.nnarchi == 'FCL':
		model = neuralnet.Net_FCL(dataset, args.hidden_sizes, args.nonlin)
	elif args.nnarchi == 'CNN':
		model = neuralnet.Net_CNN(dataset)
	else:
		model = neuralnet.resnetN(type=args.resnettype, dataset=dataset)
	model.to(device)

	#-----------------------------------#
	#----- CL-STRATEGY: ALLOCATION -----#
	#-----------------------------------#

	# Elastic Weight Consolidation (EWC)
	if isinstance(model, neuralnet.ContinualLearner):
		model.ewc_lambda = args.ewc_lambda if args.ewc else 0
		if args.ewc:
			if args.fisher_n is None or args.fisher_n < args.batch_sz:
				model.fisher_n = args.batch_sz
			else:
				model.fisher_n = args.fisher_n
			model.online = True
			model.gamma = args.gamma
			model.emp_FI = args.emp_fi


	#----------------------------------#
	#----- SEQUENCE-BASED TRAINER -----#
	#----------------------------------#

	trainer = Trainer(
		dataset = dataset,
		network = model,
		training_type = args.sequence_type,
		memory_sampling = 'reservoir sampling',
		memory_sz = args.memory_sz,
		lr=args.lr,
        momentum=0.5,
		criterion = args.loss_fn,
		optimizer = args.optimizer,
		batch_sz = args.batch_sz,
		preprocessing = False,
		device = device,
		min_visit = args.min_state_visit,
		sequence_length = args.sequence_length,
		energy_step = step,
		proba_transition = args.proba_transition,
		T = args.T,
		dynamic_T_thr = args.T_adaptive,
		split_length_list = args.split_length_list
	)

	verbose('...done', args.verbose, 0)

	rs = ResultSet()
	rs.parameters = {
		"Save root": save_root,
		"Temperature": args.T,
		"Tree Depth": dataset.depth,
		"Tree Branching": dataset.branching,
		"Flips ratio": args.data_flips_rate,
		"Sequence Length": args.sequence_length,
		"Minibatches Size": args.batch_sz,
		"Number of tests": args.test_nbr,
		"Energy Step": step,
		"Replay Memory Size": args.memory_sz,
		"Learning rate": args.lr,
		"Loss function": args.loss_fn,
		"Optimizer": args.optimizer,
		"Continual learner": "EWC" if args.ewc is True else "None",
		"Dataset": args.data_origin,
		"Random Seed": systime,
		"device_type": 'GPU' if args.cuda else 'CPU',
		"NN architecture": args.nnarchi,
		"Split total length": args.split_length_list[0],
		"Original command": str(sys.argv) # We store the original command for this set of simulations
	}
	rs.T = trainer.T
	rs.memory_sz = args.memory_sz

	if args.orig_path != "":
		# Let's check that the parameters match
		for param in [k for k in orig_checkpoint.parameters.keys() if k not in ("Random Seed", "device_type", "Original command", "Timescales")]:
			assert orig_checkpoint.parameters[param] == rs.parameters[param], "Orig checkpoint option - MISMATCH of parameter {:s}".format(param)

	train_sequenceset(trainer, args, args.block_size_shuffle_list, rs, save_root, orig_checkpoint)


if __name__ == '__main__':
	args = parser.parse_args()
	run(args)
