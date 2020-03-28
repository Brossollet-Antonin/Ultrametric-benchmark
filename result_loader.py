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

from numba import jit


hsv_unif = (0, 0, 0.15)
hsv_orig = (0, 0.9, 0.6)
hsv_tfs_orig = (0.35, 0.8, 0.6)
markers = ['o','+','x','4','s','p','P', '8', 'h', 'X']


class ResultSet:
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
	def __init__(self, sim_map_dict, dataroot, sim_struct, dataset_name, nn_config, seq_type, simset_id):
		self.sim_map_dict = sim_map_dict
		self.dataroot = dataroot
		self.sim_struct = sim_struct
		self.dataset_name = dataset_name
		self.nn_config = nn_config
		self.seq_type = seq_type
		self.simset_id = simset_id


	def load_analytics(self, load_data=False, load_atc=False, load_shuffle=True, load_htmp=False):
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
		self.lbl_htmp_orig = {}
		self.lbl_htmp_shfl = {}
	
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

		self.sim_battery_params = self.sim_map_dict[self.sim_struct][self.dataset_name][self.nn_config][self.seq_type][self.simset_id]
		folderpath = self.dataroot + '/Results/' + self.sim_struct + '/' + self.dataset_name + '/' + self.nn_config + '/' + self.sim_battery_params['folder']

		if self.seq_type in ("uniform", "ultrametric"):
			T = self.sim_battery_params["T"]
		else:
			T = 0.0

		if self.seq_type in ("uniform"):
			self.shuffle_sizes = []
		else:
			self.shuffle_sizes = self.sim_battery_params["shuffle_sizes"]

		if self.seq_type in ("random_blocks2", "ladder_blocks2"):
			block_size = self.sim_battery_params["block_size"]
		else:
			block_size = 0

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
		self.lbl_htmp_shfl[T] = {}

		if load_data:
			self.train_data_orig[T] = []
			self.train_data_shfl[T] = []

		if load_atc:
			self.atc_orig[T] = []
			self.atc_shfl[T] = []

		for simuset_path in os.listdir(folderpath):
			
			os.chdir(folderpath+'/'+simuset_path)

			with open('train_labels_orig.pickle', 'rb') as file:
				self.train_labels_orig[T].append(pickle.load(file))

			with open('distribution_train.pickle', 'rb') as file:
				self.dstr_train[T].append(pickle.load(file))
				
			with open('parameters.json', 'r') as file:
				self.params[T].append(json.load(file))

			self.eval_orig[T].append(np.load('evaluation_original.npy', allow_pickle=True))
			self.var_acc_orig[T].append(np.load('var_original_accuracy.npy'))
			self.var_pred_orig[T].append(np.load('var_original_classes_prediction.npy', allow_pickle=True))

			if load_htmp:
				with open('labels_heatmap_shfl.pickle', 'rb') as file:
					self.lbl_htmp_orig[T].append(pickle.load(file))

			if load_shuffle:
				if type(self.shuffle_sizes) is int:
					self.shuffle_sizes = [self.shuffle_sizes]

				for shuffle_sz in self.shuffle_sizes:
					if shuffle_sz not in self.train_labels_shfl[T].keys():
						self.train_labels_shfl[T][shuffle_sz] = []
						self.eval_shfl[T][shuffle_sz] = []
						self.var_acc_shfl[T][shuffle_sz] = []
						self.var_pred_shfl[T][shuffle_sz] = []

					with open('shuffle_'+str(shuffle_sz)+'/train_labels_shfl.pickle', 'rb') as file:
						self.train_labels_shfl[T][shuffle_sz].append(pickle.load(file))

					if load_htmp:
						with open('shuffle_'+str(shuffle_sz)+'/labels_heatmap_shfl.pickle', 'rb') as file:
							self.lbl_htmp_shfl[T][shuffle_sz].append(pickle.load(file))

					self.eval_shfl[T][shuffle_sz].append(np.load('shuffle_'+str(shuffle_sz)+'/evaluation_shuffled.npy', allow_pickle=True))
					self.var_acc_shfl[T][shuffle_sz].append(np.load('shuffle_'+str(shuffle_sz)+'/var_shuffle_accuracy.npy'))
					self.var_pred_shfl[T][shuffle_sz].append(np.load('shuffle_'+str(shuffle_sz)+'/var_shuffle_classes_prediction.npy', allow_pickle=True))

			if load_data:
				print("Loading data for {0:s}...".format(datapath))

				with open('train_data_orig.pickle', 'rb') as file:
					self.train_data_orig[T].append(pickle.load(file))

				if load_shuffle:
					for shuffle_sz in self.shuffle_sizes:
						self.train_data_shfl[T][shuffle_sz] = []
						with open('shuffle_'+str(shuffle_sz)+'/train_data_shfl.pickle', 'rb') as file:
							self.train_data_shfl[T][shuffle_sz].append(pickle.load(file))
					
				print("...done")

			if load_atc:
				self.atc_orig[T].append(np.load('autocorr_original.npy'))
				if load_shuffle:
					for shuffle_sz in self.shuffle_sizes:
						self.atc_shfl[T][shuffle_sz] = []
						self.atc_shfl[T][shuffle_sz].append(np.load('shuffle_'+str(shuffle_sz)+'/autocorr_shuffle.npy'))

		if not load_data:
			print("load_data set to False. Data sequences not loaded.")
		if not load_atc:
			print("load_atc set to False. Autocorrelations not loaded.")



	@jit(nopython=True)
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
			
			for seq_id, seq in enumerate(seq_list):
				print("   Original sequence {0:d}...".format(seq_id))
				for lbl_id in range(tree_l):
					locs_orig = np.array([j for j in range(len(seq)) if seq[j]==lbl_id])
					nlocs = len(locs_orig)
					locs_orig = locs_orig.reshape((nlocs, 1))
					
					locsd_mat_orig = cdist(locs_orig, locs_orig, 'cityblock')
					#     iu_ids_couples = np.array([(i,j) for j in range(20) for i in range(20*cut_id, 20*cut_id+j)])
					iu_ids = np.triu_indices(nlocs)
					iu_len = len(iu_ids[0])
					diffs = locsd_mat_orig[iu_ids].reshape((iu_len,1))
					hlocs_stat_orig = hlocs_stat_orig + np.histogram(
						diffs,
						bins=w_size,
						range=(0,w_size)
					)[0]/tree_l

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


	def lbl_history(self, T_list, shuffled_blocksz=None, strides=None):
		n_Ts = len(T_list)
		assert (n_Ts>0)
		t_explr = None

		lbls_fig = plt.figure(figsize=(18,10*n_Ts))
		lbls_axes = []

		for T_id, T in enumerate(T_list):
			if shuffled_blocksz is None:
				occur_id = random.randint(0, len(self.train_labels_orig[T])-1)
				seq = self.train_labels_orig[T][occur_id]
			else:
				occur_id = random.randint(0, len(self.train_labels_shfl[T][shuffled_blocksz])-1)
				seq = self.train_labels_shfl[T][shuffled_blocksz][occur_id]

			n_labels = len(set(seq))
			lbls_ax = plt.subplot(n_Ts, 1, 1+T_id)
			lbls_axes.append(lbls_ax)
			lbls_ax.plot(seq)

			obs_lbl_set = set()
			nobs_seq = []
			for itr_id, lbl in enumerate(seq):
				obs_lbl_set.add(lbl)
				nobs_seq.append(len(obs_lbl_set))
				if t_explr is None and len(obs_lbl_set) == n_labels:
					t_explr = itr_id

			lbls_ax.plot(nobs_seq)
			if strides is not None:
				for stride in strides:
					lbls_ax.axvline(x=stride, ymin=0, ymax=n_labels)

			ttl = 'History of labels in the original training sequence - T='+str(T)
			if t_explr:
				ttl = ttl+' - tau_asym=' + str(t_explr)
			plt.title(ttl)

		return lbls_fig, lbls_axes


def get_acc(T_list, acc_temp_orig, acc_temp_shuffled, acc_unif=None, acc_twofold_orig=None, acc_twofold_shuffled=None, seq_length=200000, n_tests=200, plot_window=None, blocks_for_shared_plots=None, var_scale=0.3, save_format='pdf'):

	n_Ts = len(T_list)
	assert (n_Ts>0)

	splt_sz = {0.0: 12500, 0.4: 1000, 0.5: 160}

	n_ticks = 10
	xtick_scale = n_tests//n_ticks
	xtick_pos = xtick_scale*np.arange((n_tests//xtick_scale)+1)
	xtick_labels = int(seq_length/((n_tests//xtick_scale)))*np.arange((n_tests//xtick_scale)+1)
	fig = plt.figure(1, figsize=(18,12*3*n_Ts))
	axes = []

	for T_id, T in enumerate(T_list):
		acc_ax = plt.subplot(3*n_Ts, 1, 1+3*T_id)
		axes.append(acc_ax)

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
				y1 = var_acc_unif - var_scale*var_acc_unif_std,
				y2 = var_acc_unif + var_scale*var_acc_unif_std,
				color = hsv_to_rgb(hsv_unif),
				alpha = 0.4
			)

		## Plotting average performance for original ultrametric sequences

		if T in acc_temp_orig.keys():
			var_acc_orig = np.mean([acc[:,0] for acc in acc_temp_orig[T]], axis=0)
			var_acc_orig_std = np.std([acc[:,0] for acc in acc_temp_orig[T]], axis=0)
			acc_ax.plot(
					var_acc_orig,
					ls = 'solid',
					color = hsv_to_rgb(hsv_orig),
					label='Ultrametric sequence, T={0:.2f} - No shuffling'.format(T)
				)
			acc_ax.fill_between(
				x = range(len(var_acc_orig)),
				y1 = var_acc_orig - var_scale*var_acc_orig_std,
				y2 = var_acc_orig + var_scale*var_acc_orig_std,
				color = hsv_to_rgb(hsv_orig),
				alpha = 0.4
			)

		if (T,1) in acc_temp_orig.keys():
			var_acc_orig = np.mean([acc[:,0] for acc in acc_temp_orig[(T,1)]], axis=0)
			var_acc_orig_std = np.std([acc[:,0] for acc in acc_temp_orig[(T,1)]], axis=0)
			acc_ax.plot(
					var_acc_orig,
					ls = 'solid',
					color = hsv_to_rgb(hsv_orig),
					label='Ultrametric sequence, T={0:.2f} - No shuffling'.format(T)
				)
			acc_ax.fill_between(
				x = range(len(var_acc_orig)),
				y1 = var_acc_orig - var_scale*var_acc_orig_std,
				y2 = var_acc_orig + var_scale*var_acc_orig_std,
				color = hsv_to_rgb(hsv_orig),
				alpha = 0.4
			)

		## Plotting average performance for shuffled ultrametric sequences
		if T in acc_temp_shuffled.keys():
			for block_id, block_sz in enumerate(sorted(acc_temp_shuffled[T].keys())):
				acc_data = acc_temp_shuffled[T][block_sz]
				hsv_shfl = tuple([0.6, 1-block_id*0.04, 0.4+block_id*0.04])

				var_acc_shfl = np.mean([acc[:,0] for acc in acc_data], axis=0)
				var_acc_shfl_std = np.std([acc[:,0] for acc in acc_data], axis=0)
				
				acc_ax.plot(
					var_acc_shfl,
					ls = '--',
					color = hsv_to_rgb(hsv_shfl),
					label='Ultrametric sequence, T={0:.2f} - Shuffled w/ block size {1:d}'.format(T, block_sz)
				)
				acc_ax.fill_between(
					x = range(len(var_acc_shfl)),
					y1 = var_acc_shfl - var_scale*var_acc_shfl_std,
					y2 = var_acc_shfl + var_scale*var_acc_shfl_std,
					color = hsv_to_rgb(hsv_shfl),
					alpha = 0.2
				)

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

		if plot_window is not None:
			plt.xlim(plot_window[0], plot_window[1])

		acc_ax_blk = plt.subplot(3*n_Ts, 1, 2+3*T_id)
		axes.append(acc_ax_blk)

		## Plotting average performance for split scenario (two-folds)

		if acc_twofold_shuffled is not None and T in acc_twofold_shuffled.keys():
			for block_id, block_sz in enumerate(sorted(acc_twofold_shuffled[T].keys())):
				acc_data = acc_twofold_shuffled[T][block_sz]
				hsv_tfs_shfl = tuple([0.35, 0.8-(block_id+1)*0.04, 0.6-(block_id+1)*0.03])
				var_acc_tfs_shfl = np.mean([acc[:,0] for acc in acc_data], axis=0)
				var_acc_tfs_shfl_std = np.std([acc[:,0] for acc in acc_data], axis=0)
				
				acc_ax_blk.plot(
					var_acc_tfs_shfl,
					ls = '--',
					color = hsv_to_rgb(hsv_tfs_shfl),
					label='Random split sequence - Shuffled w/ block size {0:d}'.format(block_sz)
				)
				acc_ax_blk.fill_between(
					x = range(len(var_acc_tfs_shfl)),
					y1 = var_acc_tfs_shfl - var_scale*var_acc_tfs_shfl_std,
					y2 = var_acc_tfs_shfl + var_scale*var_acc_tfs_shfl_std,
					color = hsv_to_rgb(hsv_tfs_shfl),
					alpha = 0.2
				)

		if acc_twofold_orig is not None and T in acc_twofold_orig.keys():
			var_acc_tfs_orig = np.mean([acc[:,0] for acc in acc_twofold_orig[T]], axis=0)
			var_acc_tfs_orig_std = np.std([acc[:,0] for acc in acc_twofold_orig[T]], axis=0)
			acc_ax_blk.plot(
				var_acc_tfs_orig,
				ls = 'solid',
				color = hsv_to_rgb(hsv_tfs_orig),
				label='Random split sequence, split size {0:d} - No shuffling'.format(splt_sz[T])
			)
			acc_ax_blk.fill_between(
				x = range(len(var_acc_tfs_orig)),
				y1 = var_acc_tfs_orig - var_scale*var_acc_tfs_orig_std,
				y2 = var_acc_tfs_orig + var_scale*var_acc_tfs_orig_std,
				color = hsv_to_rgb(hsv_tfs_orig),
				alpha = 0.4
			)

		if acc_twofold_orig is not None and (T,1) in acc_twofold_orig.keys():
			var_acc_tfs_orig = np.mean([acc[:,0] for acc in acc_twofold_orig[(T,1)]], axis=0)
			var_acc_tfs_orig_std = np.std([acc[:,0] for acc in acc_twofold_orig[(T,1)]], axis=0)
			acc_ax_blk.plot(
				var_acc_tfs_orig,
				ls = 'solid',
				color = hsv_to_rgb(hsv_tfs_orig),
				label='Random split sequence, split size {0:d} - No shuffling'.format(splt_sz[T])
			)
			acc_ax_blk.fill_between(
				x = range(len(var_acc_tfs_orig)),
				y1 = var_acc_tfs_orig - var_scale*var_acc_tfs_orig_std,
				y2 = var_acc_tfs_orig + var_scale*var_acc_tfs_orig_std,
				color = hsv_to_rgb(hsv_tfs_orig),
				alpha = 0.4
			)

		###################################################################
		
		plt.xticks(xtick_pos, xtick_labels)
		plt.title('Accuracy as a function of time for original and shuffled sequences', fontsize = 14)

		box = acc_ax_blk.get_position()
		acc_ax_blk.set_position([box.x0, box.y0 + box.height * 0.1,
						 box.width, box.height * 0.9])

		acc_ax_blk.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
					  fancybox=True, shadow=True, ncol=2,
					  prop={'size': 16})

		plt.xlabel('Iterations', fontsize=14)
		plt.ylabel('Accuracy (%)', fontsize=14)

		if plot_window is not None:
			plt.xlim(plot_window[0], plot_window[1])

		###################################################################

		acc_ax_all = plt.subplot(3*n_Ts, 1, 3+3*T_id)
		axes.append(acc_ax_all)

		if acc_unif is not None:
		## Plotting average performance for random sequences (from uniform distr)
			var_acc_unif = np.mean([acc[:,0] for acc in acc_unif], axis=0)
			var_acc_unif_std = np.std([acc[:,0] for acc in acc_unif], axis=0)
			acc_ax_all.plot(
					var_acc_unif,
					ls = 'solid',
					color = hsv_to_rgb(hsv_unif),
					label='Uniform learning'
			)

			acc_ax_all.fill_between(
				x = range(len(var_acc_unif)),
				y1 = var_acc_unif - var_scale*var_acc_unif_std,
				y2 = var_acc_unif + var_scale*var_acc_unif_std,
				color = hsv_to_rgb(hsv_unif),
				alpha = 0.4
			)

		## Plotting average performance for original ultrametric sequences

		if T in acc_temp_orig.keys():
			var_acc_orig = np.mean([acc[:,0] for acc in acc_temp_orig[T]], axis=0)
			var_acc_orig_std = np.std([acc[:,0] for acc in acc_temp_orig[T]], axis=0)
			acc_ax_all.plot(
					var_acc_orig,
					ls = 'solid',
					color = hsv_to_rgb(hsv_orig),
					label='Ultrametric sequence, T={0:.2f} - No shuffling'.format(T)
				)
			acc_ax_all.fill_between(
				x = range(len(var_acc_orig)),
				y1 = var_acc_orig - var_scale*var_acc_orig_std,
				y2 = var_acc_orig + var_scale*var_acc_orig_std,
				color = hsv_to_rgb(hsv_orig),
				alpha = 0.4
			)

		if (T,1) in acc_temp_orig.keys():
			var_acc_orig = np.mean([acc[:,0] for acc in acc_temp_orig[(T,1)]], axis=0)
			var_acc_orig_std = np.std([acc[:,0] for acc in acc_temp_orig[(T,1)]], axis=0)
			acc_ax_all.plot(
					var_acc_orig,
					ls = 'solid',
					color = hsv_to_rgb(hsv_orig),
					label='Ultrametric sequence, T={0:.2f} - No shuffling'.format(T)
				)
			acc_ax_all.fill_between(
				x = range(len(var_acc_orig)),
				y1 = var_acc_orig - var_scale*var_acc_orig_std,
				y2 = var_acc_orig + var_scale*var_acc_orig_std,
				color = hsv_to_rgb(hsv_orig),
				alpha = 0.4
			)

		## Plotting average performance for shuffled ultrametric sequences
		if T in acc_temp_shuffled.keys():
			for block_id, block_sz in enumerate(sorted(acc_temp_shuffled[T].keys())):
				acc_data = acc_temp_shuffled[T][block_sz]
				if blocks_for_shared_plots is None or block_sz in blocks_for_shared_plots:
					hsv_shfl = tuple([0.6, 1-block_id*0.04, 0.4+block_id*0.04])

					var_acc_shfl = np.mean([acc[:,0] for acc in acc_data], axis=0)
					var_acc_shfl_std = np.std([acc[:,0] for acc in acc_data], axis=0)
					
					acc_ax_all.plot(
						var_acc_shfl,
						ls = '--',
						color = hsv_to_rgb(hsv_shfl),
						label='Ultrametric sequence, T={0:.2f} - Shuffled w/ block size {1:d}'.format(T, block_sz)
					)
					acc_ax_all.fill_between(
						x = range(len(var_acc_shfl)),
						y1 = var_acc_shfl - var_scale*var_acc_shfl_std,
						y2 = var_acc_shfl + var_scale*var_acc_shfl_std,
						color = hsv_to_rgb(hsv_shfl),
						alpha = 0.2
					)

		if acc_twofold_shuffled is not None and T in acc_twofold_shuffled.keys():
			for block_id, block_sz in enumerate(sorted(acc_twofold_shuffled[T].keys())):
				acc_data = acc_twofold_shuffled[T][block_sz]
				if blocks_for_shared_plots is None or block_sz in blocks_for_shared_plots:
					hsv_tfs_shfl = tuple([0.35, 0.8-(block_id+1)*0.04, 0.6-(block_id+1)*0.03])
					var_acc_tfs_shfl = np.mean([acc[:,0] for acc in acc_data], axis=0)
					var_acc_tfs_shfl_std = np.std([acc[:,0] for acc in acc_data], axis=0)
					
					acc_ax_all.plot(
						var_acc_tfs_shfl,
						ls = '--',
						color = hsv_to_rgb(hsv_tfs_shfl),
						label='Random split sequence - Shuffled w/ block size {0:d}'.format(block_sz)
					)
					acc_ax_all.fill_between(
						x = range(len(var_acc_tfs_shfl)),
						y1 = var_acc_tfs_shfl - var_scale*var_acc_tfs_shfl_std,
						y2 = var_acc_tfs_shfl + var_scale*var_acc_tfs_shfl_std,
						color = hsv_to_rgb(hsv_tfs_shfl),
						alpha = 0.2
					)

		if acc_twofold_orig is not None and T in acc_twofold_orig.keys():
			var_acc_tfs_orig = np.mean([acc[:,0] for acc in acc_twofold_orig[T]], axis=0)
			var_acc_tfs_orig_std = np.std([acc[:,0] for acc in acc_twofold_orig[T]], axis=0)
			acc_ax_all.plot(
				var_acc_tfs_orig,
				ls = 'solid',
				color = hsv_to_rgb(hsv_tfs_orig),
				label='Random split sequence, split size {0:d} - No shuffling'.format(splt_sz[T])
			)
			acc_ax_all.fill_between(
				x = range(len(var_acc_tfs_orig)),
				y1 = var_acc_tfs_orig - var_scale*var_acc_tfs_orig_std,
				y2 = var_acc_tfs_orig + var_scale*var_acc_tfs_orig_std,
				color = hsv_to_rgb(hsv_tfs_orig),
				alpha = 0.4
			)

		if acc_twofold_orig is not None and (T,1) in acc_twofold_orig.keys():
			var_acc_tfs_orig = np.mean([acc[:,0] for acc in acc_twofold_orig[(T,1)]], axis=0)
			var_acc_tfs_orig_std = np.std([acc[:,0] for acc in acc_twofold_orig[(T,1)]], axis=0)
			acc_ax_all.plot(
				var_acc_tfs_orig,
				ls = 'solid',
				color = hsv_to_rgb(hsv_tfs_orig),
				label='Random split sequence, split size {0:d} - No shuffling'.format(splt_sz[T])
			)
			acc_ax_all.fill_between(
				x = range(len(var_acc_tfs_orig)),
				y1 = var_acc_tfs_orig - var_scale*var_acc_tfs_orig_std,
				y2 = var_acc_tfs_orig + var_scale*var_acc_tfs_orig_std,
				color = hsv_to_rgb(hsv_tfs_orig),
				alpha = 0.4
			)
			
		plt.xticks(xtick_pos, xtick_labels)
		plt.title('Accuracy as a function of time for original and shuffled sequences', fontsize = 14)

		box = acc_ax_all.get_position()
		acc_ax_all.set_position([box.x0, box.y0 + box.height * 0.1,
						 box.width, box.height * 0.9])

		acc_ax_all.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
					  fancybox=True, shadow=True, ncol=2,
					  prop={'size': 16})

		plt.xlabel('Iterations', fontsize=14)
		plt.ylabel('Accuracy (%)', fontsize=14)

		if plot_window is not None:
			plt.xlim(plot_window[0], plot_window[1])

	fig.tight_layout(pad=10.0)
	plt.savefig('out_plots_acc.'+str(save_format), format=save_format)

	return fig, axes


def get_acc_v0_markers(T_list, acc_temp_orig, acc_temp_shuffled, acc_unif=None, acc_twofold_orig=None, acc_twofold_shuffled=None, seq_length=200000, n_tests=200, discard_last_tests=0, blocks_for_shared_plots=None, var_scale=0.3, save_format='pdf'):

	n_Ts = len(T_list)
	assert (n_Ts>0)

	xtick_scale = 25
	xtick_pos = xtick_scale*np.arange((n_tests//xtick_scale)+1)
	xtick_labels = int(seq_length/((n_tests//xtick_scale)))*np.arange((n_tests//xtick_scale)+1)
	fig = plt.figure(1, figsize=(18,12*3*n_Ts))

	for T_id, T in enumerate(T_list):
		acc_ax = plt.subplot(3*n_Ts, 1, 1+3*T_id)

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
				y1 = var_acc_unif - var_scale*var_acc_unif_std,
				y2 = var_acc_unif + var_scale*var_acc_unif_std,
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
				y1 = var_acc_orig - var_scale*var_acc_orig_std,
				y2 = var_acc_orig + var_scale*var_acc_orig_std,
				color = hsv_to_rgb(hsv_orig),
				alpha = 0.4
			)

		if (T,1) in acc_temp_orig.keys():
			var_acc_orig = np.mean([acc[:,0] for acc in acc_temp_orig[(T,1)]], axis=0)
			var_acc_orig_std = np.std([acc[:,0] for acc in acc_temp_orig[(T,1)]], axis=0)
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
				y1 = var_acc_orig - var_scale*var_acc_orig_std,
				y2 = var_acc_orig + var_scale*var_acc_orig_std,
				color = hsv_to_rgb(hsv_orig),
				alpha = 0.4
			)

		## Plotting average performance for shuffled ultrametric sequences
		if T in acc_temp_shuffled.keys():
			for block_id, (block_sz, acc_data) in enumerate(acc_temp_shuffled[T].items()):
				hsv_shfl = tuple([0.6, 1-block_id*0.08, 0.4+block_id*0.08])

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
					y1 = var_acc_shfl - var_scale*var_acc_shfl_std,
					y2 = var_acc_shfl + var_scale*var_acc_shfl_std,
					color = hsv_to_rgb(hsv_shfl),
					alpha = 0.2
				)

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

		acc_ax_blk = plt.subplot(3*n_Ts, 1, 2+3*T_id)

		## Plotting average performance for split scenario (two-folds)
		if acc_twofold_orig is not None and T in acc_twofold_orig.keys():
			var_acc_tfs_orig = np.mean([acc[:,0] for acc in acc_twofold_orig[T]], axis=0)
			var_acc_tfs_orig_std = np.std([acc[:,0] for acc in acc_twofold_orig[T]], axis=0)
			acc_ax_blk.plot(
				var_acc_tfs_orig,
				marker = '.',
				markersize=10,
				ls = 'none',
				color = hsv_to_rgb(hsv_tfs_orig),
				label='T={0:.2f} - Twofold split original sequence'.format(T)
			)
			acc_ax_blk.fill_between(
				x = range(len(var_acc_tfs_orig)),
				y1 = var_acc_tfs_orig - var_scale*var_acc_tfs_orig_std,
				y2 = var_acc_tfs_orig + var_scale*var_acc_tfs_orig_std,
				color = hsv_to_rgb(hsv_tfs_orig),
				alpha = 0.4
			)

		if acc_twofold_orig is not None and (T,1) in acc_twofold_orig.keys():
			var_acc_tfs_orig = np.mean([acc[:,0] for acc in acc_twofold_orig[(T,1)]], axis=0)
			var_acc_tfs_orig_std = np.std([acc[:,0] for acc in acc_twofold_orig[(T,1)]], axis=0)
			acc_ax_blk.plot(
				var_acc_tfs_orig,
				marker = '.',
				markersize=10,
				ls = 'none',
				color = hsv_to_rgb(hsv_tfs_orig),
				label='T={0:.2f} - Twofold split original sequence'.format(T)
			)
			acc_ax_blk.fill_between(
				x = range(len(var_acc_tfs_orig)),
				y1 = var_acc_tfs_orig - var_scale*var_acc_tfs_orig_std,
				y2 = var_acc_tfs_orig + var_scale*var_acc_tfs_orig_std,
				color = hsv_to_rgb(hsv_tfs_orig),
				alpha = 0.4
			)

		if acc_twofold_shuffled is not None and T in acc_twofold_shuffled.keys():
			for block_id, (block_sz, acc_data) in enumerate(acc_twofold_shuffled[T].items()):
				
				hsv_tfs_shfl = tuple([0.35, 0.8-(block_id+1)*0.08, 0.6-(block_id+1)*0.05])
				var_acc_tfs_shfl = np.mean([acc[:,0] for acc in acc_data], axis=0)
				var_acc_tfs_shfl_std = np.std([acc[:,0] for acc in acc_data], axis=0)
				
				acc_ax_blk.plot(
					var_acc_tfs_shfl,
					marker=markers[block_id],
					markersize=10,
					ls = 'none',
					color = hsv_to_rgb(hsv_tfs_shfl),
					label='T={0:.2f}, blocksz={1:d} - Twofold split shuffled sequence'.format(T, block_sz)
				)
				acc_ax_blk.fill_between(
					x = range(len(var_acc_tfs_shfl)),
					y1 = var_acc_tfs_shfl - var_scale*var_acc_tfs_shfl_std,
					y2 = var_acc_tfs_shfl + var_scale*var_acc_tfs_shfl_std,
					color = hsv_to_rgb(hsv_tfs_shfl),
					alpha = 0.2
				)

		###################################################################
		
		plt.xticks(xtick_pos, xtick_labels)
		plt.title('Accuracy as a function of time for original and shuffled sequences', fontsize = 14)

		box = acc_ax_blk.get_position()
		acc_ax_blk.set_position([box.x0, box.y0 + box.height * 0.1,
						 box.width, box.height * 0.9])

		acc_ax_blk.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
					  fancybox=True, shadow=True, ncol=2,
					  prop={'size': 16})

		plt.xlabel('Iterations', fontsize=14)
		plt.ylabel('Accuracy (%)', fontsize=14)

		###################################################################

		acc_ax_all = plt.subplot(3*n_Ts, 1, 3+3*T_id)

		if acc_unif is not None:
		## Plotting average performance for random sequences (from uniform distr)
			var_acc_unif = np.mean([acc[:,0] for acc in acc_unif], axis=0)
			var_acc_unif_std = np.std([acc[:,0] for acc in acc_unif], axis=0)
			acc_ax_all.plot(
					var_acc_unif,
					ls = 'solid',
					color = hsv_to_rgb(hsv_unif),
					label='Uniform learning'
			)

			acc_ax_all.fill_between(
				x = range(len(var_acc_unif)),
				y1 = var_acc_unif - var_scale*var_acc_unif_std,
				y2 = var_acc_unif + var_scale*var_acc_unif_std,
				color = hsv_to_rgb(hsv_unif),
				alpha = 0.4
			)

		## Plotting average performance for original ultrametric sequences

		if T in acc_temp_orig.keys():
			var_acc_orig = np.mean([acc[:,0] for acc in acc_temp_orig[T]], axis=0)
			var_acc_orig_std = np.std([acc[:,0] for acc in acc_temp_orig[T]], axis=0)
			acc_ax_all.plot(
					var_acc_orig,
					marker = '.',
					markersize=10,
					ls = 'none',
					color = hsv_to_rgb(hsv_orig),
					label='T={0:.2f} - Original sequence'.format(T)
				)
			acc_ax_all.fill_between(
				x = range(len(var_acc_orig)),
				y1 = var_acc_orig - var_scale*var_acc_orig_std,
				y2 = var_acc_orig + var_scale*var_acc_orig_std,
				color = hsv_to_rgb(hsv_orig),
				alpha = 0.4
			)

		if (T,1) in acc_temp_orig.keys():
			var_acc_orig = np.mean([acc[:,0] for acc in acc_temp_orig[(T,1)]], axis=0)
			var_acc_orig_std = np.std([acc[:,0] for acc in acc_temp_orig[(T,1)]], axis=0)
			acc_ax_all.plot(
					var_acc_orig,
					marker = '.',
					markersize=10,
					ls = 'none',
					color = hsv_to_rgb(hsv_orig),
					label='T={0:.2f} - Original sequence'.format(T)
				)
			acc_ax_all.fill_between(
				x = range(len(var_acc_orig)),
				y1 = var_acc_orig - var_scale*var_acc_orig_std,
				y2 = var_acc_orig + var_scale*var_acc_orig_std,
				color = hsv_to_rgb(hsv_orig),
				alpha = 0.4
			)

		## Plotting average performance for shuffled ultrametric sequences
		if T in acc_temp_shuffled.keys():
			for block_id, (block_sz, acc_data) in enumerate(acc_temp_shuffled[T].items()):
				if blocks_for_shared_plots is None or block_sz in blocks_for_shared_plots:
					hsv_shfl = tuple([0.6, 1-block_id*0.08, 0.4+block_id*0.08])

					var_acc_shfl = np.mean([acc[:,0] for acc in acc_data], axis=0)
					var_acc_shfl_std = np.std([acc[:,0] for acc in acc_data], axis=0)
					
					acc_ax_all.plot(
						var_acc_shfl,
						marker=markers[block_id],
						markersize=10,
						ls = 'none',
						color = hsv_to_rgb(hsv_shfl),
						label='T={0:.2f}, blocksz={1:d} - Shuffled sequence'.format(T, block_sz)
					)
					acc_ax_all.fill_between(
						x = range(len(var_acc_shfl)),
						y1 = var_acc_shfl - var_scale*var_acc_shfl_std,
						y2 = var_acc_shfl + var_scale*var_acc_shfl_std,
						color = hsv_to_rgb(hsv_shfl),
						alpha = 0.2
					)

		if acc_twofold_orig is not None and T in acc_twofold_orig.keys():
			var_acc_tfs_orig = np.mean([acc[:,0] for acc in acc_twofold_orig[T]], axis=0)
			var_acc_tfs_orig_std = np.std([acc[:,0] for acc in acc_twofold_orig[T]], axis=0)
			acc_ax_all.plot(
				var_acc_tfs_orig,
				marker = '.',
				markersize=10,
				ls = 'none',
				color = hsv_to_rgb(hsv_tfs_orig),
				label='T={0:.2f} - Twofold split original sequence'.format(T)
			)
			acc_ax_all.fill_between(
				x = range(len(var_acc_tfs_orig)),
				y1 = var_acc_tfs_orig - var_scale*var_acc_tfs_orig_std,
				y2 = var_acc_tfs_orig + var_scale*var_acc_tfs_orig_std,
				color = hsv_to_rgb(hsv_tfs_orig),
				alpha = 0.4
			)

		if acc_twofold_orig is not None and (T,1) in acc_twofold_orig.keys():
			var_acc_tfs_orig = np.mean([acc[:,0] for acc in acc_twofold_orig[(T,1)]], axis=0)
			var_acc_tfs_orig_std = np.std([acc[:,0] for acc in acc_twofold_orig[(T,1)]], axis=0)
			acc_ax_all.plot(
				var_acc_tfs_orig,
				marker = '.',
				markersize=10,
				ls = 'none',
				color = hsv_to_rgb(hsv_tfs_orig),
				label='T={0:.2f} - Twofold split original sequence'.format(T)
			)
			acc_ax_all.fill_between(
				x = range(len(var_acc_tfs_orig)),
				y1 = var_acc_tfs_orig - var_scale*var_acc_tfs_orig_std,
				y2 = var_acc_tfs_orig + var_scale*var_acc_tfs_orig_std,
				color = hsv_to_rgb(hsv_tfs_orig),
				alpha = 0.4
			)

		if acc_twofold_shuffled is not None and T in acc_twofold_shuffled.keys():
			for block_id, (block_sz, acc_data) in enumerate(acc_twofold_shuffled[T].items()):
				if blocks_for_shared_plots is None or block_sz in blocks_for_shared_plots:
					hsv_tfs_shfl = tuple([0.35, 0.8-(block_id+1)*0.08, 0.6-(block_id+1)*0.05])
					var_acc_tfs_shfl = np.mean([acc[:,0] for acc in acc_data], axis=0)
					var_acc_tfs_shfl_std = np.std([acc[:,0] for acc in acc_data], axis=0)
					
					acc_ax_all.plot(
						var_acc_tfs_shfl,
						marker=markers[block_id],
						markersize=10,
						ls = 'none',
						color = hsv_to_rgb(hsv_tfs_shfl),
						label='T={0:.2f}, blocksz={1:d} - Twofold split shuffled sequence'.format(T, block_sz)
					)
					acc_ax_all.fill_between(
						x = range(len(var_acc_tfs_shfl)),
						y1 = var_acc_tfs_shfl - var_scale*var_acc_tfs_shfl_std,
						y2 = var_acc_tfs_shfl + var_scale*var_acc_tfs_shfl_std,
						color = hsv_to_rgb(hsv_tfs_shfl),
						alpha = 0.2
					)

		box = acc_ax_all.get_position()
		acc_ax_all.set_position([box.x0, box.y0 + box.height * 0.1,
						 box.width, box.height * 0.9])

		acc_ax_all.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
					  fancybox=True, shadow=True, ncol=2,
					  prop={'size': 16})

		plt.xlabel('Iterations', fontsize=14)
		plt.ylabel('Accuracy (%)', fontsize=14)

	fig.tight_layout(pad=10.0)
	plt.savefig('out_plots_acc.'+str(save_format), format=save_format)


def get_cf(lbl_seq, acc_orig, acc_unif, plot=False):
	nspl = len(acc_orig)
	seql = len(lbl_seq)
	t_explr = None
	n_labels = len(set(lbl_seq))

	spl_lbl_seq = [lbl_seq[k*(seql//nspl)] for k in range(nspl)]
	assert(len(spl_lbl_seq) == len(acc_orig) == len(acc_unif))

	obs_lbl_set = set()
	nobs_seq = []
	for itr_id, lbl in enumerate(spl_lbl_seq):
		obs_lbl_set.add(lbl)
		nobs_seq.append(len(obs_lbl_set))
		if t_explr is None and len(obs_lbl_set) == n_labels:
			t_explr = itr_id

	cf = (np.array(acc_unif)-np.array(acc_orig))/np.array(nobs_seq)

	if plot:
		fig = plt.figure(1, figsize=(18,12))
		cf_ax = plt.subplot(111)
		cf_ax.plot(
			cf,
			label='Forgetting score as a fct of #iteration'
		)
		
	return cf, t_explr