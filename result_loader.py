# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 10:53:11 2019

@author: Antonin
"""

import os
import pickle
import numpy as np
import json
import pdb

from copy import deepcopy
import random
from scipy.spatial.distance import cdist
import time

from matplotlib import pyplot as plt
from matplotlib.colors import hsv_to_rgb
from tqdm.notebook import tqdm


hsv_unif = (0, 0, 0.15)
hsv_orig = (0, 0.9, 0.6)
hsv_tfs_orig = (0.35, 0.8, 0.6)
markers = ['o','+','x','4','s','p','P', '8', 'h', 'X']


class ResultSet_Single:
	"""Contains the results of a simulation

	Parameters
	----------
	dataroot : str
		Path of the root where data are stored.
	datapath : str
		Path to the data.
		
	Attribute
	---------
	dataroot : str
		Path of the root where data are stored.
	datapath : str
		Path to the data.    
	
	"""
	
	def __init__(self, dataroot, datapath):
		self.dataroot = dataroot
		self.datapath = datapath
		
	def load_analytics(self, load_atc=False):
		self.train_data_orig = {}
		self.train_labels_orig = {}
		self.train_data_shfl = {}
		self.train_labels_shfl = {}
		self.dstr_train = {}
		self.params = {}
		self.atc_orig = {}
		self.atc_shfl = {}
		self.eval_orig = {}
		self.eval_shfl = {}
		self.var_acc_orig = {}
		self.var_acc_shfl = {}
		self.var_pred_orig = {}
		self.var_pred_shfl = {}
	
		os.chdir(self.dataroot+'/'+self.datapath)

		self.help = {} # will contain general information about stored analytics
			
		self.help['train_data_orig'] = """
		Type: list    Stored as: pickle
		Contains the training data inputs, for the original training sequence
		"""
		file = open('train_data_orig.pickle', 'rb')
		self.train_data_orig = pickle.load(file)
		file.close()

		self.help['train_labels_orig'] = """
		Type: list    Stored as: pickle
		Contains the training labels, cast between 0 and N_labels, for the original training sequence
		"""
		file = open('train_labels_orig.pickle', 'rb')
		self.train_labels_orig = pickle.load(file)
		file.close()

		self.help['train_data_shfl'] = """
		Type: list    Stored as: pickle
		Contains the training data inputs, for the shuffled training sequence
		"""
		file = open('train_data_shfl.pickle', 'rb')
		self.train_data_shfl = pickle.load(file)
		file.close()

		self.help['train_labels_shfl'] = """
		Type: list    Stored as: pickle
		Contains the training labels, cast between 0 and N_labels, for the shuffled training sequence
		"""
		file = open('train_labels_shfl.pickle', 'rb')
		self.train_labels_shfl = pickle.load(file)
		file.close()

		self.help['distribution_train'] = """
		Type: list    Stored as: pickle
		Counts, for each label, the corresponding number of training example
		"""
		file = open('distribution_train.pickle', 'rb')
		self.dstr_train = pickle.load(file)
		file.close()

		self.help['parameters'] = """
		Type: list    Stored as: pickle
		Counts, for each label, the corresponding number of training example
		"""
		file = open('parameters.pickle', 'rb')
		self.params = pickle.load(file)
		file.close()

		if load_atc:
			self.help['autocorr_original.npy'] = """
			Type: array    Stored as: npy
			The autocorrelation function as computed by statsmodels.tsa.stattools.act
			"""
			self.atc_orig = np.load('autocorr_original.npy')
				
			self.help['autocorr_shuffle.npy'] = """
			Type: array    Stored as: npy
			A list of autocorrelation functions, each computed on a different test sample, as computed by statsmodels.tsa.stattools.act
			"""
			self.atc_shfl = np.load('autocorr_shuffle.npy')

		self.help['diagnostic_original.npy'] = """
		Type: array    Stored as: npy
		[0] contains the average accuracy split per level of hierarchy (I don't understand the split though)
		[1][0] contains the GT pointwise to the testing sequence
		[1][1] contains the prediction pointwise to the testing sequence
		[1][2:2+N_hier-1] contains the pointwise distance between GT and prediction on the testing sequence
		"""
		self.eval_orig = np.load('diagnostic_original.npy', allow_pickle=True)
		self.eval_shfl = np.load('diagnostic_shuffle.npy', allow_pickle=True)

		self.help['var_original_accuracy.npy'] = """
		Type: array    Stored as: npy
		[0] Average accuracy over full test sequence
		[1:test_nbr] Average accuracy over each test run
		"""
		self.var_acc_orig = np.load('var_original_accuracy.npy')
		self.var_acc_shfl = np.load('var_shuffle_accuracy.npy')

		self.help['var_original_classes_prediction.npy'] = """
		Type: array    Stored as: npy
		[0:test_nbr] Contains, for each test run, the composition of the test sampl,
		as well as the progress of training as the max training ID scanned at the time of the test run
		"""
		self.var_pred_orig = np.load('var_original_classes_prediction.npy', allow_pickle=True)
		self.var_pred_shfl = np.load('var_shuffle_classes_prediction.npy', allow_pickle=True)



class ResultSet_1to1:
	"""Contains the results of a simulation

	Parameters
	----------
	dataroot : str
		Path to the folder specific to the simulation type, dataset and sequence type that we're studying
		Ex: '<project_root>/Results/1toM/MNIST_10/CNN/temporal_correlation_length200000_batches10'
	datapath : dict of form
		{ (Temp, block_sz): [
			'repo1_name',
			'repo2_name',
			...
			]
		}
		
	Attribute
	---------
	dataroot : str
		Path of the root where data are stored.
	datapath : str
		Path to the data.    
	
	"""
	def __init__(self, dataroot, datapaths):
		self.dataroot = dataroot
		self.datapaths = datapaths
		
	def load_analytics(self, load_data=False, load_atc=False, load_shuffle=True):
		print("\nLoading analytics...")

		self.train_data_orig = {}
		self.train_labels_orig = {}
		self.train_data_shfl = {}
		self.train_labels_shfl = {}
		self.dstr_train = {}
		self.params = {}
		self.atc_orig = {}
		self.atc_shfl = {}
		self.eval_orig = {}
		self.eval_shfl = {}
		self.var_acc_orig = {}
		self.var_acc_shfl = {}
		self.var_pred_orig = {}
		self.var_pred_shfl = {}
	
		self.help = {}

		self.help['train_labels_orig'] = """
				Type: list    Stored as: pickle
				Contains the training labels, cast between 0 and N_labels, for the original training sequence
				"""
		self.help['train_labels_shfl'] = """
				Type: list    Stored as: pickle
				Contains the training labels, cast between 0 and N_labels, for the shuffled training sequence
				"""
		self.help['distribution_train'] = """
				Type: list    Stored as: pickle
				Counts, for each label, the corresponding number of training example
				"""
		self.help['parameters'] = """
				Type: list    Stored as: JSON
				Refers the different parameters and hyperparameters used for this set of simulations
				"""

		self.help['diagnostic_original.npy'] = """
				Type: array    Stored as: npy
				[0] contains the average accuracy split per level of hierarchy (I don't understand the split though)
				[1][0] contains the GT pointwise to the testing sequence
				[1][1] contains the prediction pointwise to the testing sequence
				[1][2:2+N_hier-1] contains the pointwise distance between GT and prediction on the testing sequence
				"""
		self.help['var_original_accuracy.npy'] = """
				Type: array    Stored as: npy
				[0] Average accuracy over full test sequence
				[1:test_nbr] Average accuracy over each test run
				"""
		self.help['var_original_classes_prediction.npy'] = """
				Type: array    Stored as: npy
				[0:test_nbr] Contains, for each test run, the composition of the test sampl,
				as well as the progress of training as the max training ID scanned at the time of the test run
				"""

		if load_data:
			self.help['train_data_orig'] = """
					Type: list    Stored as: pickle
					Contains the training data inputs, for the original training sequence
					"""
			self.help['train_data_shfl'] = """
					Type: list    Stored as: pickle
					Contains the training data inputs, for the shuffled training sequence
					"""
		else:
			self.help['train_data_orig'] = """
					Unavailable. load_data set to False
					"""
			self.help['train_data_shfl'] = """
					Unavailable. load_data set to False
					"""

		if load_atc:
			self.help['autocorr_original.npy'] = """
					Type: array    Stored as: npy
					The autocorrelation function as computed by statsmodels.tsa.stattools.act
					"""
			self.help['autocorr_shuffle.npy'] = """
					Type: array    Stored as: npy
					A list of autocorrelation functions, each computed on a different test sample, as computed by statsmodels.tsa.stattools.act
					"""
		else:
			self.help['autocorr_original.npy'] = """
					Unavailable. load_atc set to False
					"""
			self.help['autocorr_shuffle.npy'] = """
					Unavailable. load_atc set to False
					"""

		for params, datapath_list in self.datapaths.items():
			
			self.train_labels_orig[params] = []
			self.train_labels_shfl[params] = []
			self.dstr_train[params] = []
			self.params[params] = []
			self.eval_orig[params] = []
			self.eval_shfl[params] = []
			self.var_acc_orig[params] = []
			self.var_acc_shfl[params] = []
			self.var_pred_orig[params] = []
			self.var_pred_shfl[params] = []

			if load_data:
				self.train_data_orig[params] = []
				self.train_data_shfl[params] = []

			if load_atc:
				self.atc_orig[params] = []
				self.atc_shfl[params] = []

			for datapath in datapath_list:
				os.chdir(self.dataroot+'/'+datapath)

				with open('train_labels_orig.pickle', 'rb') as file:
					self.train_labels_orig[params].append(pickle.load(file))

				with open('distribution_train.pickle', 'rb') as file:
					self.dstr_train[params].append(pickle.load(file))
				
				with open('parameters.json', 'r') as file:
					self.params[params].append(json.load(file))

				self.eval_orig[params].append(np.load('evaluation_original.npy', allow_pickle=True))
				self.var_acc_orig[params].append(np.load('var_original_accuracy.npy'))
				self.var_pred_orig[params].append(np.load('var_original_classes_prediction.npy', allow_pickle=True))

				if load_shuffle:
					with open('train_labels_shfl.pickle', 'rb') as file:
						self.train_labels_shfl[params].append(pickle.load(file))
					self.eval_shfl[params].append(np.load('evaluation_shuffled.npy', allow_pickle=True))
					self.var_acc_shfl[params].append(np.load('var_shuffle_accuracy.npy'))
					self.var_pred_shfl[params].append(np.load('var_shuffle_classes_prediction.npy', allow_pickle=True))

				if load_data:
					print("Loading data for {0:s}...".format(datapath))

					with open('train_data_orig.pickle', 'rb') as file:
						self.train_data_orig[params].append(pickle.load(file))

					with open('train_data_shfl.pickle', 'rb') as file:
						self.train_data_shfl[params].append(pickle.load(file))
					
					print("...done")

				if load_atc:
					self.atc_orig[params].append(np.load('autocorr_original.npy'))
					self.atc_shfl[params].append(np.load('autocorr_shuffle.npy'))

		if not load_data:
			print("load_data set to False. Data sequences not loaded.")
		if not load_atc:
			print("load_atc set to False. Autocorrelations not loaded.")


	def get_atc_vectorized(self, T_list, n_tests, out_filename, w_size=10000, n_omits=30):
		n_Ts = len(T_list)
		assert (n_Ts>0)

		bins_hist = range(w_size)

		plt.figure(1, figsize=(18,10*n_Ts))

		for T_id, T in enumerate(T_list):
			atc_ax = plt.subplot(n_Ts, 1, 1+T_id)

			seq_list = []
			for b in train_labels_orig.keys():
				if b[0] == 'T':
					seq_list.append(self.train_labels_orig[b])

			tree_l = max(seq_list)+1
			hlocs_stat_orig = np.zeros(w_size)
			hlocs_stat_shfl = np.zeros(w_size)
			
			print("Computing autocorrelation on {0:d} sequences".format(len(seq_list)))
			
			for seq_id, seq in tqdm(enumerate(seq_list), desc='Sequence #'):
				print("   Original sequence {0:d}...".format(seq_id))
				for lbl_id in tqdm(range(tree_l), desc='Leaf #'):
					locs_orig = np.array([j for j in range(len(seq)) if seq[j]==lbl_id])
					nlocs = len(locs_orig)
					locs_orig = locs_orig.reshape((nlocs, 1))
					
					locsd_mat_orig = cdist(locs_orig, locs_orig, 'cityblock')
					#     iu_ids_couples = np.array([(i,j) for j in range(20) for i in range(20*cut_id, 20*cut_id+j)])
					iu_ids = np.triu_indices(nlocs)
					iu_len = len(iu_ids[0])
					diffs = locsd_mat_orig[iu_ids].reshape((iu_len,1))
					hlocs_orig = hlocs_orig + np.histogram(
						diffs,
						bins=w_size,
						range=(0,w_size)
					)
					hlocs_stat_orig = hlocs_stat_orig + hlocs_orig[0]/tree_l

				print("   ...done")


			if hlocs_stat_orig[0] > 0:
				hlocs_stat_orig = hlocs_stat_orig / hlocs_stat_orig[1]
				
			bins_atc = range(w_size//2)
			atc_ax.loglog(
				bins_atc,
				hlocs_stat_orig[::2],
				marker='.',
				color = hsv_to_rgb(hsv_orig),
				ls = 'solid',
				label='T={0:.2f} - Original sequence'.format(T)
			) 
			
			hlocs_stat_shfl_list = []
			for nfig, block_sz in enumerate(self.blocks_sizes):
				print("   Block size {0:d}".format(block_sz))
				hsv_shfl = tuple([0.6, 1-nfig*0.2, 0.5+nfig*0.15])
				#plt.figure(nfig+2)
				#plt.plot(shuffleseq)
				#plt.title(block_sz)
				for seq_id, seq in enumerate(seq_list):
					shuffleseq = shuffleblocks(seq, block_sz, n_tests)
					print("       Shuffled sequence {1:d}...".format(block_sz, seq_id))
					for lbl_id in range(tree_l):
						locs_shfl = np.array([j for j in range(len(shuffleseq)) if  shuffleseq[j]==lbl_id])
						nlocs = len(locs_shfl)
						locs_shfl = locs_shfl.reshape((nlocs, 1))
						locsd_mat_shfl = cdist(locs_shfl, locs_shfl, 'cityblock')     
						iu_ids = np.triu_indices(nlocs)
						bins = range(w_size)
						hlocs_shfl = np.bincount(
							locsd_mat_shfl[iu_ids].reshape((int(nlocs*(nlocs+1)/2),1))
						)
						# hlocs_shfl = np.histogram(
						#     locsd_mat_shfl[iu_ids].reshape((int(nlocs*(nlocs+1)/2),1)),
						#     bins=w_size,
						#     range=(0, w_size)
						# )
						
						hlocs_stat_shfl = hlocs_stat_shfl + hlocs_shfl[0]/tree_l
					print("       ...done")
				
				if hlocs_stat_shfl[0] > 0:
					hlocs_stat_shfl = hlocs_stat_shfl / hlocs_stat_shfl[0]
				  
				atc_ax.loglog(
					bins_atc,
					hlocs_stat_shfl[::2],
					marker = markers[nfig],
					ls = 'solid',
					color = hsv_to_rgb(hsv_shfl),
					label='T={0:.2f} - Shuffled with blocksz={1:d}'.format(T, block_sz),
					alpha=0.5) 
				hlocs_stat_shfl_list.append(hlocs_stat_shfl)
				
			plt.title('Autocorrelation of training sequence')
			plt.xlabel('t, number of iterations /2', fontsize=12)
			plt.ylabel('A(t)', fontsize=14)

			atc_ax.set_position([box.x0, box.y0 + box.height * 0.1,
				 box.width, box.height * 0.9])
			atc_ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
			  fancybox=True, shadow=True, ncol=2,
			  prop={'size': 16})
			
			plt.savefig(
				fname=filename+'.pdf',
				format='pdf'
			)
				
			return hlocs_stat_orig, hlocs_stat_shfl_list


	def lbl_history(self, T_list):
		n_Ts = len(T_list)
		assert (n_Ts>0)

		lbls_fig = plt.figure(figsize=(18,10*n_Ts))

		for T_id, T in enumerate(T_list):
			lbls_ax = plt.subplot(n_Ts, 1, 1+T_id)
			lbls_ax.plot(self.train_labels_orig[(T,1)][0])
			ttl = 'History of labels in the original training sequence - T='+str(T)
			plt.title(ttl)


class ResultSet_1toM:
	"""Contains the results of a simulation

	Parameters
	----------
	dataroot : str
		Path to the folder specific to the simulation type, dataset and sequence type that we're studying
		Ex: '<project_root>/Results/1toM/MNIST_10/CNN/temporal_correlation_length200000_batches10'
	datapath : dict of form
		{ Temp: [
			('repo1_name', block_sz_tuple1),
			('repo2_name', block_sz_tuple2)
			]
		}
		
	Attribute
	---------
	dataroot : str
		Path of the root where data are stored.
	datapath : str
		Path to the data.    
	
	"""
	def __init__(self, dataroot, datapaths):
		self.dataroot = dataroot
		self.datapaths = datapaths
		self.block_sizes = set()
		
	def load_analytics(self, load_data=False, load_atc=False, load_shuffle=True):
		print("\nLoading analytics...")

		self.train_data_orig = {}
		self.train_labels_orig = {}
		self.train_data_shfl = {}
		self.train_labels_shfl = {}
		self.dstr_train = {}
		self.params = {}
		self.atc_orig = {}
		self.atc_shfl = {}
		self.eval_orig = {}
		self.eval_shfl = {}
		self.var_acc_orig = {}
		self.var_acc_shfl = {}
		self.var_pred_orig = {}
		self.var_pred_shfl = {}
	
		self.help = {}

		self.help['train_labels_orig'] = """
				Type: list    Stored as: pickle
				Contains the training labels, cast between 0 and N_labels, for the original training sequence
				"""
		self.help['train_labels_shfl'] = """
				Type: list    Stored as: pickle
				Contains the training labels, cast between 0 and N_labels, for the shuffled training sequence
				"""
		self.help['distribution_train'] = """
				Type: list    Stored as: pickle
				Counts, for each label, the corresponding number of training example
				"""
		self.help['parameters'] = """
				Type: list    Stored as: JSON
				Refers the different parameters and hyperparameters used for this set of simulations
				"""

		self.help['diagnostic_original.npy'] = """
				Type: array    Stored as: npy
				[0] contains the average accuracy split per level of hierarchy (I don't understand the split though)
				[1][0] contains the GT pointwise to the testing sequence
				[1][1] contains the prediction pointwise to the testing sequence
				[1][2:2+N_hier-1] contains the pointwise distance between GT and prediction on the testing sequence
				"""
		self.help['var_original_accuracy.npy'] = """
				Type: array    Stored as: npy
				[0] Average accuracy over full test sequence
				[1:test_nbr] Average accuracy over each test run
				"""
		self.help['var_original_classes_prediction.npy'] = """
				Type: array    Stored as: npy
				[0:test_nbr] Contains, for each test run, the composition of the test sampl,
				as well as the progress of training as the max training ID scanned at the time of the test run
				"""

		if load_data:
			self.help['train_data_orig'] = """
					Type: list    Stored as: pickle
					Contains the training data inputs, for the original training sequence
					"""
			self.help['train_data_shfl'] = """
					Type: list    Stored as: pickle
					Contains the training data inputs, for the shuffled training sequence
					"""
		else:
			self.help['train_data_orig'] = """
					Unavailable. load_data set to False
					"""
			self.help['train_data_shfl'] = """
					Unavailable. load_data set to False
					"""

		if load_atc:
			self.help['autocorr_original.npy'] = """
					Type: array    Stored as: npy
					The autocorrelation function as computed by statsmodels.tsa.stattools.act
					"""
			self.help['autocorr_shuffle.npy'] = """
					Type: array    Stored as: npy
					A list of autocorrelation functions, each computed on a different test sample, as computed by statsmodels.tsa.stattools.act
					"""
		else:
			self.help['autocorr_original.npy'] = """
					Unavailable. load_atc set to False
					"""
			self.help['autocorr_shuffle.npy'] = """
					Unavailable. load_atc set to False
					"""

		for T, datapath_list in self.datapaths.items():
			
			self.train_labels_orig[T] = []
			self.train_labels_shfl[T] = {}
			self.dstr_train[T] = []
			self.params[T] = []
			self.eval_orig[T] = []
			self.eval_shfl[T] = {}
			self.var_acc_orig[T] = []
			self.var_acc_shfl[T] = {}
			self.var_pred_orig[T] = []
			self.var_pred_shfl[T] = {}

			if load_data:
				self.train_data_orig[T] = []
				self.train_data_shfl[T] = []

			if load_atc:
				self.atc_orig[T] = []
				self.atc_shfl[T] = []

			for datapath, block_sizes in datapath_list:
				os.chdir(self.dataroot+'/'+datapath)

				with open('train_labels_orig.pickle', 'rb') as file:
					self.train_labels_orig[T].append(pickle.load(file))

				with open('distribution_train.pickle', 'rb') as file:
					self.dstr_train[T].append(pickle.load(file))
				
				with open('parameters.json', 'r') as file:
					self.params[T].append(json.load(file))

				self.eval_orig[T].append(np.load('evaluation_original.npy', allow_pickle=True))
				self.var_acc_orig[T].append(np.load('var_original_accuracy.npy'))
				self.var_pred_orig[T].append(np.load('var_original_classes_prediction.npy', allow_pickle=True))

				if load_shuffle:
					for block_sz in block_sizes:
						if block_sz not in self.block_sizes:
							self.block_sizes.add(block_sz)
						self.train_labels_shfl[T][block_sz] = []
						self.eval_shfl[T][block_sz] = []
						self.var_acc_shfl[T][block_sz] = []
						self.var_pred_shfl[T][block_sz] = []
						with open('shuffle_'+str(block_sz)+'/train_labels_shfl.pickle', 'rb') as file:
							self.train_labels_shfl[T][block_sz].append(pickle.load(file))
						with open('shuffle_'+str(block_sz)+'/train_labels_shfl.pickle', 'rb') as file:
							self.train_labels_shfl[T][block_sz].append(pickle.load(file))
						self.eval_shfl[T][block_sz].append(np.load('shuffle_'+str(block_sz)+'/evaluation_shuffled.npy', allow_pickle=True))
						self.var_acc_shfl[T][block_sz].append(np.load('shuffle_'+str(block_sz)+'/var_shuffle_accuracy.npy'))
						self.var_pred_shfl[T][block_sz].append(np.load('shuffle_'+str(block_sz)+'/var_shuffle_classes_prediction.npy', allow_pickle=True))

				if load_data:
					print("Loading data for {0:s}...".format(datapath))

					with open('train_data_orig.pickle', 'rb') as file:
						self.train_data_orig[T].append(pickle.load(file))

					if load_shuffle:
						for block_sz in block_sizes:
							self.train_data_shfl[T][block_sz] = []
							with open('shuffle_'+str(block_sz)+'/train_data_shfl.pickle', 'rb') as file:
								self.train_data_shfl[T][block_sz].append(pickle.load(file))
					
					print("...done")

				if load_atc:
					self.atc_orig[T].append(np.load('autocorr_original.npy'))
					if load_shuffle:
						for block_sz in block_sizes:
							self.atc_shfl[T][block_sz] = []
							self.atc_shfl[T][block_sz].append(np.load('shuffle_'+str(block_sz)+'/autocorr_shuffle.npy'))

		if not load_data:
			print("load_data set to False. Data sequences not loaded.")
		if not load_atc:
			print("load_atc set to False. Autocorrelations not loaded.")


	def get_atc(self, T_list, n_tests, out_filename, w_size=10000, n_omits=30):
		n_Ts = len(T_list)
		assert (n_Ts>0)

		bins_hist = range(w_size)

		plt.figure(1, figsize=(18,10*n_Ts))

		for T_id, T in enumerate(T_list):
			atc_ax = plt.subplot(n_Ts, 1, 1+T_id)

			seq_list = self.train_labels_orig[T]
			tree_l = max(seq_list)+1
			hlocs_stat_orig = np.zeros(w_size)
			hlocs_stat_shfl = np.zeros(w_size)
			
			print("Computing autocorrelation on {0:d} sequences".format(len(seq_list)))
			
			for seq_id, seq in tqdm(enumerate(seq_list), desc='Sequence #'):
				print("   Original sequence {0:d}...".format(seq_id))
				for lbl_id in tqdm(range(tree_l), desc='Leaf #'):
					locs_orig = np.array([j for j in range(len(seq)) if seq[j]==lbl_id])
					nlocs = len(locs_orig)
					locs_orig = locs_orig.reshape((nlocs, 1))
					
					locsd_mat_orig = cdist(locs_orig, locs_orig, 'cityblock')
					#     iu_ids_couples = np.array([(i,j) for j in range(20) for i in range(20*cut_id, 20*cut_id+j)])
					iu_ids = np.triu_indices(nlocs)
					iu_len = len(iu_ids[0])
					diffs = locsd_mat_orig[iu_ids].reshape((iu_len,1))
					hlocs_orig = hlocs_orig + np.histogram(
						diffs,
						bins=w_size,
						range=(0,w_size)
					)
					hlocs_stat_orig = hlocs_stat_orig + hlocs_orig[0]/tree_l

				print("   ...done")


			if hlocs_stat_orig[0] > 0:
				hlocs_stat_orig = hlocs_stat_orig / hlocs_stat_orig[1]
				
			bins_atc = range(w_size//2)
			atc_ax.loglog(
				bins_atc,
				hlocs_stat_orig[::2],
				marker='.',
				color = hsv_to_rgb(hsv_orig),
				ls = 'solid',
				label='T={0:.2f} - Original sequence'.format(T)
			) 
			
			hlocs_stat_shfl_list = []
			for nfig, block_sz in enumerate(self.blocks_sizes):
				print("   Block size {0:d}".format(block_sz))
				hsv_shfl = tuple([0.6, 1-nfig*0.2, 0.5+nfig*0.15])
				#plt.figure(nfig+2)
				#plt.plot(shuffleseq)
				#plt.title(block_sz)
				for seq_id, seq in enumerate(seq_list):
					shuffleseq = shuffleblocks(seq, block_sz, n_tests)
					print("       Shuffled sequence {1:d}...".format(block_sz, seq_id))
					for lbl_id in range(tree_l):
						locs_shfl = np.array([j for j in range(len(shuffleseq)) if  shuffleseq[j]==lbl_id])
						nlocs = len(locs_shfl)
						locs_shfl = locs_shfl.reshape((nlocs, 1))
						locsd_mat_shfl = cdist(locs_shfl, locs_shfl, 'cityblock')     
						iu_ids = np.triu_indices(nlocs)
						bins = range(w_size)
						hlocs_shfl = np.bincount(
							locsd_mat_shfl[iu_ids].reshape((int(nlocs*(nlocs+1)/2),1))
						)
						# hlocs_shfl = np.histogram(
						#     locsd_mat_shfl[iu_ids].reshape((int(nlocs*(nlocs+1)/2),1)),
						#     bins=w_size,
						#     range=(0, w_size)
						# )
						
						hlocs_stat_shfl = hlocs_stat_shfl + hlocs_shfl[0]/tree_l
					print("       ...done")
				
				if hlocs_stat_shfl[0] > 0:
					hlocs_stat_shfl = hlocs_stat_shfl / hlocs_stat_shfl[0]
				  
				atc_ax.loglog(
					bins_atc,
					hlocs_stat_shfl[::2],
					marker = markers[nfig],
					ls = 'solid',
					color = hsv_to_rgb(hsv_shfl),
					label='T={0:.2f} - Shuffled with blocksz={1:d}'.format(T, block_sz),
					alpha=0.5) 
				hlocs_stat_shfl_list.append(hlocs_stat_shfl)
				
			plt.title('Autocorrelation of training sequence')
			plt.xlabel('t, number of iterations /2', fontsize=12)
			plt.ylabel('A(t)', fontsize=14)

			atc_ax.set_position([box.x0, box.y0 + box.height * 0.1,
				 box.width, box.height * 0.9])
			atc_ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
			  fancybox=True, shadow=True, ncol=2,
			  prop={'size': 16})
			
			plt.savefig(
				fname=filename+'.pdf',
				format='pdf'
			)
				
			return hlocs_stat_orig, hlocs_stat_shfl_list


	def lbl_history(self, T_list):
		n_Ts = len(T_list)
		assert (n_Ts>0)

		lbls_fig = plt.figure(figsize=(18,10*n_Ts))

		for T_id, T in enumerate(T_list):
			lbls_ax = plt.subplot(n_Ts, 1, 1+T_id)
			lbls_ax.plot(self.train_labels_orig[T][0])
			ttl = 'History of labels in the original training sequence - T='+str(T)
			plt.title(ttl)


def get_acc(T_list, acc_temp_orig, acc_temp_shuffled, acc_unif=None, acc_twofold_orig=None, acc_twofold_shuffled=None, seq_length=200000, n_tests=200):

	n_Ts = len(T_list)
	assert (n_Ts>0)

	xtick_scale = 25
	xtick_pos = xtick_scale*np.arange((n_tests//xtick_scale)+1)
	xtick_labels = int(seq_length/((n_tests//xtick_scale)))*np.arange((n_tests//xtick_scale)+1)
	fig = plt.figure(1, figsize=(18,12*n_Ts))

	for T_id, T in enumerate(T_list):
		acc_ax = plt.subplot(n_Ts, 1, 1+T_id)

		if acc_unif is not None:
		## Plotting average performance for random sequences (from uniform distr)
			var_acc_unif = np.mean([acc[:,0] for acc in acc_unif], axis=0)
			var_acc_unif_std = np.std([acc[:,0] for acc in acc_unif], axis=0)
			acc_ax.plot(
					var_acc_unif,
					ls = 'solid',
					color = hsv_to_rgb(hsv_unif),
					label='Uniform learning'
			)

			acc_ax.fill_between(
				x = range(len(var_acc_unif)),
				y1 = var_acc_unif - var_acc_unif_std,
				y2 = var_acc_unif + var_acc_unif_std,
				color = hsv_to_rgb(hsv_unif),
				alpha = 0.4
			)

		## Plotting average performance for original ultrametric sequences

		if T in acc_temp_orig.keys():
			var_acc_orig = np.mean([acc[:,0] for acc in acc_temp_orig[T]], axis=0)
			var_acc_orig_std = np.std([acc[:,0] for acc in acc_temp_orig[T]], axis=0)
			acc_ax.plot(
					var_acc_orig,
					marker = '.',
					markersize=10,
					ls = 'none',
					color = hsv_to_rgb(hsv_orig),
					label='T={0:.2f} - Original sequence'.format(T)
				)
			acc_ax.fill_between(
				x = range(len(var_acc_orig)),
				y1 = var_acc_orig - 0.3*var_acc_orig_std,
				y2 = var_acc_orig + 0.3*var_acc_orig_std,
				color = hsv_to_rgb(hsv_orig),
				alpha = 0.4
			)

		## Plotting average performance for shuffled ultrametric sequences
		if T in acc_temp_shuffled.keys():
			for block_id, (block_sz, acc_data) in enumerate(acc_temp_shuffled[T].items()):
				hsv_shfl = tuple([0.6, 1-block_id*0.12, 0.4+block_id*0.12])

				var_acc_shfl = np.mean([acc[:,0] for acc in acc_data], axis=0)
				var_acc_shfl_std = np.std([acc[:,0] for acc in acc_data], axis=0)
				
				acc_ax.plot(
					var_acc_shfl,
					marker=markers[block_id],
					markersize=10,
					ls = 'none',
					color = hsv_to_rgb(hsv_shfl),
					label='T={0:.2f}, blocksz={1:d} - Shuffled sequence'.format(T, block_sz)
				)
				acc_ax.fill_between(
					x = range(len(var_acc_shfl)),
					y1 = var_acc_shfl - 0.3*var_acc_shfl_std,
					y2 = var_acc_shfl + 0.3*var_acc_shfl_std,
					color = hsv_to_rgb(hsv_shfl),
					alpha = 0.2
				)

		## Plotting average performance for split scenario (two-folds)
		if acc_twofold_orig is not None and T in acc_twofold_orig.keys():
			var_acc_tfs_orig = np.mean([acc[:,0] for acc in acc_twofold_orig[T]], axis=0)
			var_acc_tfs_orig_std = np.std([acc[:,0] for acc in acc_twofold_orig[T]], axis=0)
			acc_ax.plot(
				var_acc_tfs_orig,
				marker = '.',
				markersize=10,
				ls = 'none',
				color = hsv_to_rgb(hsv_tfs_orig),
				label='T={0:.2f} - Twofold split original sequence'.format(T)
			)
			acc_ax.fill_between(
				x = range(len(var_acc_tfs_orig)),
				y1 = var_acc_tfs_orig - var_acc_tfs_orig_std,
				y2 = var_acc_tfs_orig + var_acc_tfs_orig_std,
				color = hsv_to_rgb(hsv_tfs_orig),
				alpha = 0.4
			)

		if acc_twofold_shuffled is not None and T in acc_twofold_shuffled.keys():
			for block_id, (block_sz, acc_data) in enumerate(acc_twofold_shuffled[T].items()):
				
				hsv_tfs_shfl = tuple([0.35, 0.8-(block_id+1)*0.12, 0.6-(block_id+1)*0.12])
				var_acc_tfs_shfl = np.mean([acc[:,0] for acc in acc_data], axis=0)
				var_acc_tfs_shfl_std = np.std([acc[:,0] for acc in acc_data], axis=0)
				
				acc_ax.plot(
					var_acc_tfs_shfl,
					marker=markers[block_id],
					markersize=10,
					ls = 'none',
					color = hsv_to_rgb(hsv_tfs_shfl),
					label='T={0:.2f}, blocksz={1:d} - Twofold split shuffled sequence'.format(T, block_sz)
				)
				acc_ax.fill_between(
					x = range(len(var_acc_tfs_shfl)),
					y1 = var_acc_tfs_shfl - var_acc_tfs_shfl_std,
					y2 = var_acc_tfs_shfl + var_acc_tfs_shfl_std,
					color = hsv_to_rgb(hsv_tfs_shfl),
					alpha = 0.2
				)

		###################################################################
		
		plt.xticks(xtick_pos, xtick_labels)
		plt.title('Accuracy as a function of time for original and shuffled sequences', fontsize = 14)

		box = acc_ax.get_position()
		acc_ax.set_position([box.x0, box.y0 + box.height * 0.1,
						 box.width, box.height * 0.9])

		acc_ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
					  fancybox=True, shadow=True, ncol=2,
					  prop={'size': 16})

		plt.xlabel('Iterations', fontsize=14)
		plt.ylabel('Accuracy (%)', fontsize=14)

	fig.tight_layout(pad=10.0)
	plt.savefig('out_plots_acc.pdf', format='pdf')