# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 10:53:11 2019

@author: Antonin
"""

import os
import errno
import pickle
import numpy as np
import json
import pdb
import re
import getpass
import datetime

import utils

from copy import deepcopy
import random
from scipy.spatial.distance import cdist
import time
import matplotlib

from matplotlib import pyplot as plt
from matplotlib.colors import hsv_to_rgb
import seaborn as sns
from tqdm.notebook import tqdm

from numba import jit

hsv_unif = (0, 0, 0.15)
markers = ['o','+','x','4','s','p','P', '8', 'h', 'X']
conf_fscore={
    0.9: 1.64,
    0.95: 1.96,
    0.98: 2.33,
    0.99: 2.58,
}

paths = utils.get_project_paths()

class ResultSet:
	"""Contains the results of a simulation

	Attribute
	---------
	set_descr: str
		A description of the result set
		Used for legends in the plots that will be generated.
	sim_map_dict: dict
		The simu_mapping file parsed into a Python dictionnary. This dict maps simulation configuration to folders containing all simulations for that configuration.
	dataroot : str
		Path to the folder specific to the simulation type, dataset and sequence type that we're studying
		Ex: '<project_root>/Results/1toM/MNIST_10/CNN/temporal_correlation_length200000_batches10'
	sim_struct: str
		Data structure generated for the simulations. If simulations were generated after February 15th 2020, this will be '1toM' (default) 
	dataset_name: str
		Identifies the data that you're dealing with.
		'MNIST' or 'artificial' or perhaps 'CIFAR'. Do not mention the number of classes here.
	nn_config: str
		Short identifier for the structure of neural network that you have used for the simulations.
		Strct: <archi_name><archi_width_param>
		Ex: 'FCL20' for a set of 3 Fully Connected Layers with mid layer width 20.
		The meaning of <archi_width_param> depends on the architecture type and is hard-coded at the moment.
	seq_type: str
		Uniquely identifies the generation strategy for the sequence of labels that your network will be learning on.
		See simu_mapping dict for details and examples of seq_types. Please name your seq_type in a clear and meaningful way.
	simset_id: int or float
		For given sim_struct, dataset_name, nn_config and seq_type, each simulation set is identified by the value of an hyperparameter.
		For ultrametric sequences, this is temperature (float)
		For random_blocks2, this is the size of the shuffling block (int)
	hsv_orig: tuple (h,s,v)
		HSV description of the color that will be used for plots for the original sequence, meaning the sequence that has not been shuffled.
	hsv_shfl_list: list((h,s,v))
		A list of HSV descriptions of the colors that will be used for plots for the shuffled sequences.
		This list should be at least as long as the maximum number of shuffling block sizes that you will perform, as each curve will need to be identified visually.
	train_labels_orig: list of arrays
		Contains the training labels, cast between 0 and N_labels, for the original training sequence
		Originally stored as: pickle
	train_labels_shfl: dict of list of arrays
		Contains the training labels, cast between 0 and N_labels, for the shuffled training sequence
		Dict mapping each shuffling block size to a list of sequences, one per simulation
		Originally stored as: pickle
	distribution_train: list
		Counts, for each label, the corresponding number of training example
		Originally stored as: pickle
	parameters: dict
		Refers the different parameters and hyperparameters used for this set of simulations
		Originally stored as: JSON
	diagnostic_original
		[0] contains the average accuracy split per level of hierarchy (I don't understand the split though)
		[1][0] contains the GT pointwise to the testing sequence
		[1][1] contains the prediction pointwise to the testing sequence
		[1][2:2+N_hier-1] contains the pointwise distance between GT and prediction on the testing sequence
		Originally stored as: npy
	var_original_accuracy: array
		[0] Average accuracy over full test sequence
		[1:test_nbr] Average accuracy over each test run
		Originally stored as: npy
	var_original_classes_prediction: array
		[0:test_nbr] Contains, for each test run, the composition of the test sample,
		as well as the progress of training as the max training ID scanned at the time of the test run
		Originally stored as: npy
	autocorr_original: array
		The autocorrelation function as computed by statsmodels.tsa.stattools.act
		Only available if load_atc set to True in self.load_analytics()
		Originally tored as: npy
	autocorr_shuffle: array
		A list of autocorrelation functions, each computed on a different test sample, as computed by statsmodels.tsa.stattools.act
		Originally stored as: npy
	"""

	def __init__(self, rs_name, rs_descr, sim_map_dict, dataset_name, nn_config, seq_type, simset_id, hue=0.5, sim_struct='1toM'):
		"""Instanciates the ResultSet, identified by a set of hyperparameters

		Parameters
		----------
		set_descr: str
			A unique name given to this dataset, describing its specifities
			Used for legends in the plots that will be generated.
		sim_map_dict: dict
			The simu_mapping file parsed into a Python dictionnary. This dict maps simulation configuration to folders containing all simulations for that configuration.
		sim_struct: str
			Data structure generated for the simulations. If simulations were generated after February 15th 2020, this will be '1toM' (default) 
		dataset_name: str
			Identifies the data that you're dealing with.
			'MNIST' or 'artificial' or perhaps 'CIFAR'. Do not mention the number of classes here.
		nn_config: str
			Short identifier for the structure of neural network that you have used for the simulations.
			Strct: <archi_name><archi_width_param>
			Ex: 'FCL20' for a set of 3 Fully Connected Layers with mid layer width 20.
			The meaning of <archi_width_param> depends on the architecture type and is hard-coded at the moment.
		seq_type: str
			Uniquely identifies the generation strategy for the sequence of labels that your network will be learning on.
			See simu_mapping dict for details and examples of seq_types. Please name your seq_type in a clear and meaningful way.
		simset_id: int or float
			For given sim_struct, dataset_name, nn_config and seq_type, each simulation set is identified by the value of an hyperparameter.
			For ultrametric sequences, this is temperature (float)
			For random_blocks2, this is the size of the shuffling block (int)
		hue: int
			Hue used for the color of the family of plots generated from this simulation. Default 0.5
		uniform: bool
			True if the sequence the simulation is done on unifrom sequence
		"""
		
		self.name = rs_name
		self.descr = rs_descr
		self.sim_map_dict = sim_map_dict
		self.sim_struct = sim_struct
		self.dataset_name = dataset_name
		self.nn_config = nn_config
		self.seq_type = seq_type
		self.simset_id = simset_id
		self.hue = hue
		self.uniform = "uniform" in self.seq_type


	def load_analytics(self, load_shuffle=True, load_atc=False, load_htmp=False):
		"""
		Loads the data from sim_map_dict for self hyperparameters

		Parameters
		----------
		load_shuffle: bool
			If True, loads the data for shuffled sequences.
			True by default, but needs to be False for uniform sequences
		load_atc: bool
			If True, precomputed autocorrelation function will be loaded
			Defaults to False, and right now no autocorrelation precomputation is produced (too slow to produce)
		load_htmp: bool
			If True, a heatmap that provides the average exploration of each class as a function of iteration will be produced.


		"""
		print("Loading result set for {0:s}...".format(self.descr))
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


		self.sim_battery_params = self.sim_map_dict[self.sim_struct][self.dataset_name][self.nn_config][self.seq_type][self.simset_id]
		folderpath = "{0:s}{1:s}/{2:s}/{3:s}/{4:s}/".format(
			paths['simus'],
			self.sim_struct,
			self.dataset_name,
			self.nn_config,
			self.sim_battery_params['folder']
		)

		if ("uniform" in self.seq_type) or ("ultrametric" in self.seq_type):
			T = self.sim_battery_params["T"]
		else:
			T = 0.0

		if self.seq_type in ("uniform"):
			self.shuffle_sizes = []
		else:
			self.shuffle_sizes = self.sim_battery_params["shuffle_sizes"]

		if ("random_blocks2" in self.seq_type) or ("ladder_blocks2" in self.seq_type):
			block_size = self.sim_battery_params["block_size"]
		else:
			block_size = 0

		self.train_labels_orig = []
		self.train_labels_shfl = {}
		self.dstr_train = []
		self.params = []
		self.eval_orig = []
		self.eval_shfl = {}
		self.var_acc_orig = []
		self.var_acc_shfl = {}
		self.var_pred_orig = []
		self.var_pred_shfl = {}
		self.lbl_htmp_shfl = {}

		if load_atc:
			self.atc_orig = []
			self.atc_shfl = []

		if self.seq_type in ("random_blocks2", "ladder_blocks2") and 'sleb' not in getpass.getuser(): # Patch temporaire,
				# A terme, il faut qu'on trouve pourquoi la structure de nos folders diffère
			block_folders = os.listdir(folderpath)
			block_folder = [blocks for blocks in block_folders if re.search(rf'{block_size}\b', blocks)][0]
			folderpath = folderpath+'/'+block_folder+'/'
		
		for simuset in os.listdir(folderpath):
			simuset_path = folderpath+'/'+simuset+'/'

			with open(simuset_path+'train_labels_orig.pickle', 'rb') as file:
				self.train_labels_orig.append(pickle.load(file))

			with open(simuset_path+'distribution_train.pickle', 'rb') as file:
				self.dstr_train.append(pickle.load(file))

			with open(simuset_path+'parameters.json', 'r') as file:
				self.params.append(json.load(file))

			self.eval_orig.append(np.load(simuset_path+'evaluation_original.npy', allow_pickle=True))
			self.var_acc_orig.append(np.load(simuset_path+'var_original_accuracy.npy'))
			self.var_pred_orig.append(np.load(simuset_path+'var_original_classes_prediction.npy', allow_pickle=True))

			if load_htmp:
				with open(simuset_path+'labels_heatmap_shfl.pickle', 'rb') as file:
					self.lbl_htmp_orig.append(pickle.load(file))

			if load_shuffle:
				if type(self.shuffle_sizes) is int:
					self.shuffle_sizes = [self.shuffle_sizes]

				for shuffle_sz in self.shuffle_sizes:
					if shuffle_sz not in self.train_labels_shfl.keys():
						self.train_labels_shfl[shuffle_sz] = []
						self.eval_shfl[shuffle_sz] = []
						self.var_acc_shfl[shuffle_sz] = []
						self.var_pred_shfl[shuffle_sz] = []

					with open(simuset_path+'shuffle_'+str(shuffle_sz)+'/train_labels_shfl.pickle', 'rb') as file:
						self.train_labels_shfl[shuffle_sz].append(pickle.load(file))

					if load_htmp:
						with open(simuset_path+'shuffle_'+str(shuffle_sz)+'/labels_heatmap_shfl.pickle', 'rb') as file:
							self.lbl_htmp_shfl[shuffle_sz].append(pickle.load(file))

					self.eval_shfl[shuffle_sz].append(np.load(simuset_path+'shuffle_'+str(shuffle_sz)+'/evaluation_shuffled.npy', allow_pickle=True))
					self.var_acc_shfl[shuffle_sz].append(np.load(simuset_path+'shuffle_'+str(shuffle_sz)+'/var_shuffle_accuracy.npy'))
					self.var_pred_shfl[shuffle_sz].append(np.load(simuset_path+'shuffle_'+str(shuffle_sz)+'/var_shuffle_classes_prediction.npy', allow_pickle=True))


			if load_atc:
				self.atc_orig.append(np.load(simuset_path+'autocorr_original.npy'))
				if load_shuffle:
					for shuffle_sz in self.shuffle_sizes:
						self.atc_shfl[shuffle_sz] = []
						self.atc_shfl[shuffle_sz].append(np.load(simuset_path+'shuffle_'+str(shuffle_sz)+'/autocorr_shuffle.npy'))

		if load_atc:
			print("load_atc set to True. Autocorrelations loaded.")

		self.set_hsv(self.hue, self.uniform)


	def set_hsv(self, hue=0.5, uniform=False):
		l_shfl = len(self.shuffle_sizes) if not uniform else 1
		self.hsv_orig = [hue, 1, 0.7] if not uniform else [0, 0, 0.15]
		sat_stride = 1/l_shfl
		value_stride = 0.5/l_shfl
		self.hsv_shfl_list = [[hue, 1-sat_stride*shfl_id, 0.45+value_stride*shfl_id] for shfl_id in range(l_shfl)]


	def lbl_history(self, shuffled_blocksz=None, strides=None):
		"""
		Plots an example of sequence as label=f(iter)

		Parameters
		----------
		shuffled_blocksz: int
			If not None, will pick one of the simulated sequences shuffled with the specified shuffling block size
		strides: list
			If not None, must be a list of int/float with the position of vertical bars to be plotted on top of the figure
		"""
		t_explr = None

		lbls_fig, lbls_ax = plt.subplots(figsize=(18,10))

		if shuffled_blocksz is None:
			occur_id = random.randint(0, len(self.train_labels_orig)-1)
			seq = self.train_labels_orig[occur_id]
		else:
			occur_id = random.randint(0, len(self.train_labels_shfl[shuffled_blocksz])-1)
			seq = self.train_labels_shfl[shuffled_blocksz][occur_id]

		n_labels = len(set(seq))
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

		ttl = 'History of labels in the original training sequence'
		if t_explr:
			ttl = ttl+' - tau_asym=' + str(t_explr)
		plt.title(ttl)

		return lbls_fig, lbls_axes


	def lbl_distrib(self, max_iter:int=None, shuffled_blocksz=None, filter_perc:float=2, cumulative=False, save_formats=None, xaxis_n_ticks=10):
		"""
		Plots the distribution of relative representation of labels for the sequence, averaged over all sequences of the simulation set.
		If shuffle_blocksz is left None, the set of original sequences will be used for plotting.
		Otherwise, the set of sequences shuffled with the specified bklock size will be used. 

		Parameters
		----------
		max_iter: int
			If not None, will only plot distributions up to iteration max_iter
		shuffled_blocksz: int
			If not None, will pick one of the simulated sequences shuffled with the specified shuffling block size
		filter_perc: float
			size of the gaussian filter used for distribution smooting, in percentage of the total sequence length
		cumulative: bool
			if True, the relative share over all
		save_formats: list(str):
			if not None, must contain the list of format the distribution figures should be output as
		"""
		from scipy.ndimage import gaussian_filter1d

		seq_length = self.params[0]["Sequence Length"]
		n_tests = self.params[0]["Number of tests"]
		n_classes = self.params[0]["Tree Branching"]**self.params[0]["Tree Depth"]
		distrib = np.zeros(shape=(seq_length, n_classes))

		fig, (heatmap_ax, distr_ax) = plt.subplots(nrows=2, ncols=1, figsize=(18,20))

		if shuffled_blocksz is None:
			seq_set = self.train_labels_orig
		else:
			seq_set = self.train_labels_shfl[shuffled_blocksz]

		n_seqs = len(seq_set)

		for seq in seq_set:
			if len(seq) > seq_length:
				seq = seq[0:seq_length]
			distrib[range(seq_length), seq] += 1/n_seqs

		# Optionally, cumulative summing over iterations
		if cumulative:
			norm = np.arange(1, 1+float(seq_length))
			distrib = np.divide(
				np.cumsum(distrib, axis=0).transpose(),
				norm
				).transpose()

		# Gaussian smoothing
		if max_iter is not None:
			filter_sz = filter_perc*max_iter/100
		else:
			filter_sz = filter_perc*seq_length/100
		filt_distrib = np.copy(distrib)
		for cl_id in range(n_classes):
			filt_distrib[:, cl_id] = gaussian_filter1d(distrib[:, cl_id], filter_sz)

		# Plot heatmap of different labels
		xtick_scale = n_tests//xaxis_n_ticks
		xtick_pos = int(seq_length/((n_tests//xtick_scale)))*np.arange((n_tests//xtick_scale)+1)

		if max_iter is not None:
			sns.heatmap(
				np.transpose(filt_distrib[0:max_iter,:]),
				ax = heatmap_ax
				)
		else:
			sns.heatmap(
				np.transpose(filt_distrib),
				ax = heatmap_ax
				)

		heatmap_ax.set_xticks(xtick_pos)

		heatmap_ax.set_xlabel('Iteration')
		heatmap_ax.set_ylabel('Class label')

		# Plot all distributions on one plot
		for cl_id in range(n_classes):
			hsv = [cl_id/n_classes, 1, 0.7]
			if max_iter is not None:
				distr_ax.plot(
					filt_distrib[0:max_iter, cl_id],
					color = hsv_to_rgb(hsv)
				)
			else:
				distr_ax.plot(
					filt_distrib[:, cl_id],
					color = hsv_to_rgb(hsv)
				)
			
		if save_formats is not None:
			for fmt in save_formats:
				out_filepath = os.path.join(
					paths['plots'],
					"labels_distrib/distrib_{setname:s}_{maxiter:s}_{blocksz:s}_{filt:s}_{cum:s}.{fmt:s}".format(
						setname = self.name,
						maxiter = 'maxiter{:d}'.format(max_iter) if max_iter is not None else 'fullseq',
						blocksz = 'blocksz{:d}'.format(shuffled_blocksz) if shuffled_blocksz is not None else 'orig',
						filt = 'filtered_{:.0f}%'.format(filter_perc) if filter_perc is not None else 'unfiltered',
						cum = 'cumulative' if cumulative else 'instantaneous',
						fmt = fmt
					)
				)
				if not os.path.exists(os.path.dirname(out_filepath)):
				    try:
				        os.makedirs(os.path.dirname(out_filepath))
				    except OSError as exc: # Guard against race condition
				        if exc.errno != errno.EEXIST:
				            raise

				plt.savefig(out_filepath, format=fmt)

		return fig, heatmap_ax, distr_ax
			

	@jit(nopython=True)
	def get_atc(self, T_list, n_tests, out_filename, w_size=10000):
		"""
		Computes the (avg) autocorrelation function of a the sequences in a simulation set
		Note: will only work in JIT compilation. Running this on JIT has not been fully tested yet, need more work.
		Right now we're using Matlab to compute autocorrelation functions.

		Parameters
		----------
		T_list : list of floats
			List of temperatures for which simulations were ran and are registered in this simulation set
		n_tests : ?
		w_size : int
			size of the window that is used for computing autocorrelation.
			To reduce computation time, we only look a given number of elements forward on the sequence, corresponding to w_size
		"""
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
				fname=paths['plots']+filename+'.pdf',
				format='pdf'
			)

			return hlocs_stat_orig, hlocs_stat_shfl_list
	  

def make_perfplot(rs, blocks, ax, plt_confinter=False):
	"""
	Generates a plot of classification accuracy as a function of number of iteration for a given result set. 

	Parameters
	----------
	rs : ResultSet
		the ResultSet object containing the simulations for which we wish to plot accuracy = f(iter)
	blocks : list(int)
		a list of the shuffle block sizes that were used in the simulations
		Note that all simulations should use the same shuffle block sizes
	ax : matplotlib axes object
		the axes to plot on
	plt_confinter: bool
		if True, confidence intervals for confidence level 95% will be printed.
		Defaults to False
		Note that because of the stochastic nature of sequence generation, the time of discovery of the different classes of the problem varies a lot, resulting in artificially exacerbated variance.

	"""

	### ORIGINAL ###
	n_orig = len(rs.var_acc_orig)
	var_acc_orig = np.mean([acc[:,0] for acc in rs.var_acc_orig], axis=0)
	var_acc_orig_std = np.std([acc[:, 0] for acc in rs.var_acc_orig], axis=0)
	x_labels = rs.var_acc_orig[0][:,1]
	ax.plot(
			x_labels, var_acc_orig,
			ls = 'solid',
			color = hsv_to_rgb(rs.hsv_orig),
			label=rs.descr+' - No shuffling'
		)

	if plt_confinter:
		ax.fill_between(
			x = x_labels,
			y1 = np.maximum(0, var_acc_orig - conf_fscore[0.95]*np.sqrt(var_acc_orig*(100-var_acc_orig)/n_orig)),
			y2 = np.minimum(var_acc_orig + conf_fscore[0.95]*np.sqrt(var_acc_orig*(100-var_acc_orig)/n_orig), 100),
			color = hsv_to_rgb(rs.hsv_orig),
			alpha = 0.4
		)

	### SHUFFLES ###
	if blocks is None and hasattr(rs, 'var_acc_shfl'):
		blocks = sorted(rs.var_acc_shfl.keys())

	for block_id, block_sz in enumerate(blocks):
		if block_sz not in rs.var_acc_shfl.keys():
			continue
		n_shfl = len(rs.var_acc_shfl[block_sz])
		acc_data = rs.var_acc_shfl[block_sz]

		var_acc_shfl = np.mean([acc[:,0] for acc in acc_data], axis=0)
		var_acc_shfl_std = np.std([acc[:,0] for acc in acc_data], axis=0)
		ax.plot(
			x_labels, var_acc_shfl,
			ls = '--',
			color = hsv_to_rgb(rs.hsv_shfl_list[block_id]),
			label=rs.descr+' - Shuffled w/ block size {0:d}'.format(block_sz)
		)

		if plt_confinter:
			ax.fill_between(
				x = x_labels,
				y1 = np.maximum(0, var_acc_shfl - conf_fscore[0.95]*np.sqrt(var_acc_shfl*(100-var_acc_shfl)/n_shfl)),
				y2 = np.minimum(var_acc_shfl + conf_fscore[0.95]*np.sqrt(var_acc_shfl*(100-var_acc_shfl)/n_shfl), 100),
				color = hsv_to_rgb(rs.hsv_shfl_list[block_id]),
				alpha = 0.2
			)


def format_perf_plot(ax, title, xtick_pos, xtick_labels, plot_window=None):
	"""
	Formats an accuracy plot to fit paper conventions and standards. 

	Parameters
	----------
	ax : matplotlib axes object
		the axes to perform formatting on
	title : str
		title of the plot
	xtick_pos : list(float)
		positions to save x-axis ticks on
	xtick_labels: list(str)
		labels to put on the x-axis ticks which positions are set
	plot_window: tuple like (x_min, x_max)
		tuple of extrema positions to use for formating x-axis
	"""
	box = ax.get_position()
	ax.set_position([box.x0, box.y0 + box.height * 0.1,
					 box.width, box.height * 0.9])

	ax.legend()
	ax.set_xlabel('Iterations')
	ax.set_ylabel('Accuracy (%)')

	#ax.set_xticks(xtick_pos)
	#ax.set_xticklabels(xtick_labels)

	if plot_window is not None:
		ax.set_xlim(plot_window[0], plot_window[1])


def get_acc(
	rs, rs_altr=None, rs_unif=None,
	seq_length=300000, n_tests=300, plot_window=None, blocks=None,
	blocks_for_shared_plots=None, plt_confinter=False, n_ticks=10, save_formats=None
	):
	"""
	Creates accuracy plots from three ResultSet objects:
	- a main result set
	- an alternative result set (could be a different generation strategy or a different hyperparameter like bit-flipping ratio)
	- the baseline ResultSet with uniform generation stategy
	Will generate three plots:
	1) main result set vs baseline uniform, with shuffling block sizes in argument "blocks"
	2) alternative result set vs baseline uniform, with shuffling block sizes in argument "blocks"
	3) all three result set, with shuffling blocks in argument "blocks_for_shared_plots" only

	Parameters
	----------
	rs : ResultSet
		main result set
	rs_altr : ResultSet
		alternative result set
	rs_unif : ResultSet
		baseline result set with uniform generation strategy
	seq_length: int
		length of the sequences
	n_tests: int
		number of tests during learning, corresponing to the number of time while moving forward on the sequence where we stop training and evaluate performance on a test set.
	plot_window: tuple like (x_min, x_max)
		if not None, will restrict plotting to a subset of iterations with (x_min, x_max)
	blocks: list(int)
		list of shuffle block sizes that were used for shuffling. Those will be plotted in dotted line in order to assess the effect of shuffling.
		Note that the same set of shuffle block sizes must be used by the main result set and the alternative result set
	blocks_for_shared_plots: list(int)
		list of shuffle block sizes that were used for shuffling. Those will be plotted in dotted line in the third plot.
	plt_confinter: bool
		if True, confidence intervals for confidence level 95% will be printed.
		Defaults to False
		Note that because of the stochastic nature of sequence generation, the time of discovery of the different classes of the problem varies a lot, resulting in artificially exacerbated variance.
	n_ticks: int
		number of ticks to use on x-axis. Ticks will be uniformly distributed between 0 and seq_length
	save_formats: list(str)
		if not None, plots will be saved in the provided formats
	"""
	xtick_scale = n_tests//n_ticks
	xtick_pos = xtick_scale*np.arange((n_tests//xtick_scale)+1)
	xtick_labels = int(seq_length/((n_tests//xtick_scale)))*np.arange((n_tests//xtick_scale)+1)

	n_plots = 1 + 2*int(rs_altr is not None)
	fig = plt.figure(1, figsize=(18,12*n_plots))
	axes = []

	acc_ax = plt.subplot(n_plots, 1, 1)
	axes.append(acc_ax)

	### 1) UNIFORM + ORIGINAL ###
	if rs_unif is not None:
		make_perfplot(rs_unif, blocks=blocks, ax=acc_ax, plt_confinter=plt_confinter)

	make_perfplot(rs, blocks=blocks, ax=acc_ax, plt_confinter=plt_confinter)

	format_perf_plot(acc_ax, "Accuracy as a function of time for original and shuffled sequence - " + rs.descr, xtick_pos, xtick_labels, plot_window)


	### 2) UNIFORM + ALTERNATIVE ###
	if rs_altr is not None:
		acc_ax_altr = plt.subplot(n_plots, 1, 2)
		axes.append(acc_ax_altr)

		if rs_unif is not None:
			make_perfplot(rs_unif, blocks=blocks, ax=acc_ax_altr, plt_confinter=plt_confinter)

		make_perfplot(rs_altr, blocks=blocks, ax=acc_ax_altr, plt_confinter=plt_confinter)

		format_perf_plot(acc_ax_altr, "Accuracy as a function of time for original and shuffled sequence - " + rs_altr.descr, xtick_pos, xtick_labels, plot_window)


	### 3) ALL SCENARIOS, REDUCED NUMBER OF BLOCKS ###
	if rs_altr is not None:
		acc_ax_all = plt.subplot(n_plots, 1, 3)
		axes.append(acc_ax_all)

		if blocks_for_shared_plots is None and blocks is not None:
			blocks_for_shared_plots = blocks

		if rs_unif is not None:
			make_perfplot(rs_unif, blocks=blocks_for_shared_plots, ax=acc_ax_all, plt_confinter=plt_confinter)

		make_perfplot(rs_altr, blocks=blocks_for_shared_plots, ax=acc_ax_all, plt_confinter=plt_confinter)
		make_perfplot(rs, blocks=blocks_for_shared_plots, ax=acc_ax_all, plt_confinter=plt_confinter)


		format_perf_plot(acc_ax_all, "Comparative accuracy as a function of time for different scenarios", xtick_pos, xtick_labels, plot_window)

	fig.tight_layout(pad=10.0)


	if save_formats is not None:
		for fmt in save_formats:
			out_filepath = os.path.join(
				paths['plots'],
				"accuracy/accuracy_main{mainsetname:s}_altr{altersetname:s}_unif{unifsetname:s}_{date:s}.{fmt:s}".format(
					mainsetname = rs.name,
					altersetname = rs_altr.name,
					unifsetname = rs_unif.name,
					date = datetime.datetime.now().strftime("%Y%m%d"),
					fmt = fmt
				)
			)

			if not os.path.exists(os.path.dirname(out_filepath)):
			    try:
			        os.makedirs(os.path.dirname(out_filepath))
			    except OSError as exc: # Guard against race condition
			        if exc.errno != errno.EEXIST:
			            raise

			plt.savefig(out_filepath, format=fmt)

	return fig, axes



def get_raw_cf(lbl_seq, acc_orig, acc_unif, n_labels, plot=False):
	"""
	Computes the RPLF score based on a sequence, the accuracy curve of the original sequence and the accuracy score of the fully shuffled (block size 1) sequence

	Parameters
	----------
	lbl_seq : list(int)
		sequence of labels for this simulation. This is used to determine the timeframe of exploration of the different classes.
	acc_orig : list(float)
		list of accuracy scores (%) for the original sequence
	acc_unif : list(float)
		list of accuracy scores (%) for the sequence shuffled with a block size of 1, meaning that on learning, the algorithm is only limited by the incomplete exploration of classes as well as the imbalance in the sequence up to this point.
		The imbalance is the same as in the original sequence at any point in time. 
	plot: bool
		if True, will plot the RPLF as a function of the number of iterations since the first full exploration

	Returns
	-------
	cf: list
		list of the RPLF scores from the time of first full exploration of the classes.
		Therefore the scores are aligned from the first iteration that all classes were explored, onward.
		If at least one class is missing from the sequence, then None
	t_explr: int
		The first iteration at which all classes were explored.
		If at least one class is missing from the sequence, then None
	"""
	nspl = len(acc_orig)
	seql = len(lbl_seq)
	t_explr = None

	obs_lbl_set = set()
	nobs_seq = []
	for itr_id, lbl in enumerate(lbl_seq):
		obs_lbl_set.add(lbl)
		nobs_seq.append(len(obs_lbl_set))
		if t_explr is None and len(obs_lbl_set) == n_labels:
			t_explr = itr_id//(seql//nspl)

	spl_nobs_seq = [nobs_seq[k*(seql//nspl)] for k in range(nspl)]
	try:
		assert(len(spl_nobs_seq) == len(acc_orig) == len(acc_unif))
	except:
		pdb.set_trace()

	cf = (np.array(acc_unif)-np.array(acc_orig))/np.array(spl_nobs_seq)

	if plot:
		fig = plt.figure(1, figsize=(18,12))
		cf_ax = plt.subplot(111)
		cf_ax.plot(
			cf,
			label='Forgetting score as a fct of #iteration'
		)

	return cf, t_explr


def get_cf_stats(rs, blocks, ax, var_scale=1, plt_confinter=False):
	avg_cf = {}
	avg_cf_std = {}
	init_cf = {}
	init_cf_std = {}

	n_labels = rs.params[0]["Tree Branching"]**rs.params[0]["Tree Depth"]

	for seq_id, seq in enumerate(rs.train_labels_orig):
		t_explr = []
		cf = []
		_cf, _t_explr = get_raw_cf(
			seq,
			rs.var_acc_orig[seq_id][:,0],
			rs.var_acc_shfl[1][seq_id][:,0],
			n_labels
		)
		if _t_explr is not None:
			cf_aligned = np.concatenate([
				np.array(_cf[_t_explr:]),
				np.zeros(_t_explr)
			])
			cf.append(cf_aligned)
			t_explr.append(_t_explr)

	if len(cf) > 0:
		cf_mean = np.mean(
			np.stack(cf, axis=1),
			axis=1
		)
		cf_std = np.std(
			np.stack(cf, axis=1),
			axis=1
		)
		ax.plot(
			cf_mean,
			color = hsv_to_rgb(rs.hsv_orig),
			ls = 'solid',
			label = rs.descr + ' - Original sequence'
		)

		if plt_confinter:
			ax.fill_between(
				x = range(len(cf_mean)),
				y1 = cf_mean - var_scale*cf_std,
				y2 = cf_mean + var_scale*cf_std,
				color = hsv_to_rgb(rs.hsv_orig),
				alpha = 0.4
			)

		avg_cf[0] = np.mean(cf_mean)
		avg_cf_std[0] = np.mean(cf_std)
		init_cf[0] = cf_mean[0]
		init_cf_std[0] = cf_std[0]

	for block_id, block_sz in enumerate(blocks):
		assert block_sz in rs.train_labels_shfl.keys()
		t_explr = []
		cf = []
		for seq_id, seq in enumerate(rs.train_labels_shfl[block_sz]):
			_cf, _t_explr = get_raw_cf(
				seq,
				rs.var_acc_shfl[block_sz][seq_id][:,0],
				rs.var_acc_shfl[1][seq_id][:,0],
				n_labels
			)
			if _t_explr is not None:
				cf_aligned = np.concatenate([
					np.array(_cf[_t_explr:]),
					np.zeros(_t_explr)
				])
				cf.append(cf_aligned)
				t_explr.append(_t_explr)

		if len(cf) > 0:
			cf_mean = np.mean(
				np.stack(cf, axis=1),
				axis=1
			)
			cf_std = np.std(
				np.stack(cf, axis=1),
				axis=1
			)
			ax.plot(
				cf_mean,
				color = hsv_to_rgb(rs.hsv_shfl_list[block_id]),
				ls = '--',
				label = rs.descr + ' - Shuffled w/ block size {0:d}'.format(block_sz)
			)

			if plt_confinter:
				ax.fill_between(
					x = range(len(cf_mean)),
					y1 = cf_mean - var_scale*cf_std,
					y2 = cf_mean + var_scale*cf_std,
					color = hsv_to_rgb(rs.hsv_shfl_list[block_id]),
					alpha = 0.4
				)

			avg_cf[block_sz] = np.mean(cf_mean)
			avg_cf_std[block_sz] = np.mean(cf_std)
			init_cf[block_sz] = cf_mean[0]
			init_cf_std[block_sz] = cf_std[0]

	return avg_cf, avg_cf_std, init_cf, init_cf_std


def load_cf_set(
	rs, blocks=None,
	seq_length=300000, n_tests=300,
	xtick_scale=25, plt_confinter=False, save_formats=None
	):
# def get_cf(lbl_seq, acc_orig, acc_unif, plot=False):
	xtick_pos = xtick_scale*np.arange((n_tests//xtick_scale)+1)
	xtick_labels = int(seq_length/((n_tests//xtick_scale)))*np.arange((n_tests//xtick_scale)+1)

	fig_cfscore = plt.figure(figsize=(18,12))
	cf_ax = plt.subplot(1,1,1)

	avg_cf, avg_cf_std, init_cf, init_cf_std = get_cf_stats(rs, blocks, ax=cf_ax, plt_confinter=plt_confinter)

	cf_ax.set_xticks(xtick_pos)
	cf_ax.set_xticklabels(xtick_labels)
	cf_ax.set_title('Per-label loss in classification performance from the moment all classes are explored', fontsize = 18)

	box = cf_ax.get_position()
	cf_ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])

	cf_ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=2, prop={'size': 16})

	cf_ax.set_xlabel('Iterations', fontsize=14)
	cf_ax.set_ylabel('Accuracy loss from CF (%)', fontsize=14)

	fig_cfscore.tight_layout(pad=10.0)
	if save_formats is not None:
		for fmt in save_formats:
			out_filepath = os.path.join(
				paths['plots'],
				"CFscore/history/cfscoreshist_{setname:s}_{date:s}.{fmt:s}".format(
					setname = rs.name,
					date = datetime.datetime.now().strftime("%Y%m%d"),
					fmt = fmt
				)
			)

			if not os.path.exists(os.path.dirname(out_filepath)):
			    try:
			        os.makedirs(os.path.dirname(out_filepath))
			    except OSError as exc: # Guard against race condition
			        if exc.errno != errno.EEXIST:
			            raise

			plt.savefig(out_filepath, format=fmt)

	return avg_cf, avg_cf_std, init_cf, init_cf_std


def plot_cf_profile(cf_sets, method='mean', x_origpos=8.5e4, vline_pos=8.2e4, xlog=False, ylog=False, var_scale=1, save_formats=None):
	"""
	Produces plots of the CF score as a function of 
	"""
	fig_mean_cfs = plt.figure(figsize=(18,12))
	ax_mean_cfs = plt.subplot(111)

	for cf_set in cf_sets:
		rs = cf_set['rs']

		if method=='mean':
			cf = cf_set['avg_cf']
			cf_std = cf_set['avg_cf_std']
		else:
			cf = cf_set['init_cf']
			cf_std = cf_set['init_cf_std']

		xtick_pos = [k for k in sorted(cf.keys()) if k>0]
		xtick_labels = [str(k) for k in sorted(cf.keys()) if k>0]

		ax_mean_cfs.plot(
			xtick_pos,
			[cf[k] for k in sorted(cf.keys()) if k>0],
			ls = 'solid',
			linewidth=3,
			marker = '+',
			markersize = 15,
			markeredgewidth = 3,
			color = hsv_to_rgb(rs.hsv_orig),
			label = rs.descr
		)
		ax_mean_cfs.set_xticks(xtick_pos, xtick_labels)
		ax_mean_cfs.fill_between(
			x = xtick_pos,
			y1 = [cf[k] - var_scale*cf_std[k] for k in sorted(cf.keys()) if k>0],
			y2 = [cf[k] + var_scale*cf_std[k] for k in sorted(cf.keys()) if k>0],
			color = hsv_to_rgb(rs.hsv_orig),
			alpha = 0.2
		)

		ax_mean_cfs.plot(
			x_origpos,
			cf[0],
			marker = '+',
			markersize = 20,
			markeredgewidth = 4,
			color = hsv_to_rgb(rs.hsv_orig)
		)
		ax_mean_cfs.plot(
			x_origpos,
			cf[0] - var_scale*cf_std[0],
			marker = '_',
			markersize = 10,
			markeredgewidth = 4,
			color = hsv_to_rgb(rs.hsv_orig)
		)
		ax_mean_cfs.plot(
			x_origpos,
			cf[0] + var_scale*cf_std[0],
			marker = '_',
			markersize = 10,
			markeredgewidth = 4,
			color = hsv_to_rgb(rs.hsv_orig)
		)
		ax_mean_cfs.plot(
			[x_origpos, x_origpos],
			[cf[0] - var_scale*cf_std[0], cf[0] + var_scale*cf_std[0]],
			color = hsv_to_rgb(rs.hsv_orig),
			alpha = 0.2
		)

		ax_mean_cfs.hlines(y=cf[0], xmin=0, xmax=1.1*x_origpos, linestyles=':', linewidth=3, color = hsv_to_rgb(rs.hsv_orig))

	# Plot formatting for figure 4 of paper

	#xtick_pos = [k for k in xtick_pos] + [x_origpos]
	#xtick_labels = [str(k) for k in xtick_pos] + [25000]
	ax_mean_cfs.set_xticks(xtick_pos)
	ax_mean_cfs.set_xticklabels(xtick_labels)

	ax_mean_cfs.set_title('Per-label loss in classification performance as a function of shuffle block size', fontsize = 18)

	ax_mean_cfs.legend(fancybox=True, shadow=True, prop={'size': 16})

	ax_mean_cfs.set_xlabel('Shuffle length')
	ax_mean_cfs.set_ylabel('Average per-label loss from CF (%)')

	fig_mean_cfs.tight_layout(pad=10.0)


	if xlog:
		ax_mean_cfs.set_xscale("log")
	if ylog:
		ax_mean_cfs.set_yscale("log")

	for tick in ax_mean_cfs.xaxis.get_major_ticks():
		tick.label.set_rotation('vertical')

	ax_mean_cfs.set_xlim(0, 1.1 * x_origpos)
	ylim = ax_mean_cfs.get_ylim()
	ax_mean_cfs.vlines(x=vline_pos, ymin=ylim[0], ymax=ylim[1])
	#ax_mean_cfs.set_ylim(-5, 12)

	# Saving figure
	if save_formats is not None:
		for fmt in save_formats:
			out_filepath = os.path.join(
				paths['plots'],
				"CFscore/profile/cfscoresprofile_{cf_dictname:s}_{method:s}_{date:s}.{fmt:s}".format(
					cf_dictname = utils.nameof(cf_sets),
					method = method,
					date = datetime.datetime.now().strftime("%Y%m%d"),
					fmt = fmt
				)
			)

			if not os.path.exists(os.path.dirname(out_filepath)):
			    try:
			        os.makedirs(os.path.dirname(out_filepath))
			    except OSError as exc: # Guard against race condition
			        if exc.errno != errno.EEXIST:
			            raise

			plt.savefig(out_filepath, format=fmt)


def format_paper(fig_width=13.2, fig_height=9, size=15, line_width=1.5,
				axis_line_width=1.0, tick_size=12, tick_label_size=20,
				label_pad=4, legend_loc='lower right'):
	def cm2inch(x): return x/2.54
	fig_height = cm2inch(fig_height)
	fig_width = cm2inch(fig_width)
	rcParams = matplotlib.rcParams

	rcParams["figure.figsize"] = [fig_width, fig_height]   #default is [6.4, 4.8]
	rcParams["font.sans-serif"] = "Tahoma"
	rcParams["font.size"] = size
	rcParams["legend.fontsize"] = size
	rcParams["legend.frameon"] = False
	rcParams["legend.loc"] = legend_loc
	rcParams["axes.labelsize"] = size
	rcParams["xtick.labelsize"] = tick_label_size
	rcParams["ytick.labelsize"] = tick_label_size
	rcParams["xtick.major.size"] = tick_size
	rcParams["ytick.major.size"] = tick_size
	rcParams["axes.titlesize"] = 0 # no title for paper
	rcParams["axes.labelpad"] = label_pad  # default is 4.0
	rcParams["axes.linewidth"] = axis_line_width
	rcParams["lines.linewidth"] = line_width
	rcParams["xtick.direction"] = "in"
	rcParams["ytick.direction"] = "in"
	rcParams["lines.antialiased"] = True
	rcParams["savefig.dpi"] = 320


def add_letter_figure(ax, letter, fontsize=15):
	ax.text(-0.1, 1.15, letter, transform=ax.transAxes, fontsize=fontsize
			, fontweight='bold', va='top', ha='right')