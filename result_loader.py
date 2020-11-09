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
import math
from pathlib import Path

import utils

from copy import deepcopy
import random
from scipy.spatial.distance import cdist
import scipy.stats
import time
import matplotlib

from matplotlib import pyplot as plt
import matplotlib.ticker
from matplotlib import rcParams, rcParamsDefault
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
default_figset_name = 'Common_Figures'

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
	hsv_shfl_dict: list((h,s,v))
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

	def __init__(self, rs_name, rs_descr, sim_map_dict, dataset_name, nn_config, seq_type, seq_length, simset_id, hue=0.5, sim_struct='1toM'):
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
		self.seq_length = seq_length
		self.simset_id = simset_id
		self.hue = hue
		self.uniform = "uniform" in self.seq_type


	def load_analytics(self, load_shuffle=True, load_evals=False, load_atc=False, load_htmp=False, hue=None):
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
		if hue is not None:
			self.hue = hue

		print("Loading result set for {0:s}...".format(self.descr))
		self.train_data_orig = {}
		self.train_labels_orig = {}
		self.train_data_shfl = {}
		self.train_labels_shfl = {}
		self.dstr_train = {}
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

		self.block_sizes = []

		self.sim_battery_params = self.sim_map_dict[self.sim_struct][self.dataset_name][self.nn_config][self.seq_type][self.simset_id]
		folderpath = Path(paths['simus'])
		# folderpath.joinpath(self.sim_struct,
		# 	self.dataset_name,
		# 	self.nn_config,
		# 	self.sim_battery_params['folder']
		# )
		folderpath = folderpath / self.dataset_name / self.nn_config / self.sim_battery_params['folder']
		with open(folderpath / 'parameters.json', 'r') as param_file:
			self.params = json.load(param_file)

		self.n_tests = self.params["Number of tests"]
		self.res_temp = self.seq_length // self.n_tests
		self.n_classes = self.params["Tree Branching"]**self.params["Tree Depth"]

		if ("uniform" in self.seq_type) or ("ultrametric" in self.seq_type):
			T = self.sim_battery_params["T"]
		else:
			T = 0.0

		if "uniform" in self.seq_type:
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
		self.classes_templates = []
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
		
		self.n_seqs = {}

		simuset_list = [simuset for simuset in folderpath.iterdir() if (folderpath/simuset).is_dir()]
		for simuset in simuset_list:
			simuset_path = folderpath / simuset

			with open(simuset_path / 'train_labels_orig.pickle', 'rb') as file:
				self.train_labels_orig.append(pickle.load(file))

			with open(simuset_path / 'distribution_train.pickle', 'rb') as file:
				self.dstr_train.append(pickle.load(file))

			self.classes_templates.append(np.load(simuset_path / 'classes_templates.npy', allow_pickle=True))

			if load_evals:
				self.eval_orig.append(np.load(simuset_path / 'evaluation_original.npy', allow_pickle=True))
			self.var_acc_orig.append(np.load(simuset_path / 'var_original_accuracy.npy', allow_pickle=True))
			self.var_pred_orig.append(np.load(simuset_path / 'var_original_classes_prediction.npy', allow_pickle=True))

			if load_htmp:
				with open(simuset_path / 'labels_heatmap_shfl.pickle', 'rb') as file:
					self.lbl_htmp_orig.append(pickle.load(file))

			if load_shuffle:
				if type(self.shuffle_sizes) is int:
					self.shuffle_sizes = [self.shuffle_sizes]

				for shuffle_sz in self.shuffle_sizes:
					if shuffle_sz not in self.block_sizes:
						self.block_sizes.append(shuffle_sz)
						self.train_labels_shfl[shuffle_sz] = []
						self.eval_shfl[shuffle_sz] = []
						self.var_acc_shfl[shuffle_sz] = []
						self.var_pred_shfl[shuffle_sz] = []

					with open(simuset_path / 'shuffle_{:d}'.format(shuffle_sz) / 'train_labels_shfl.pickle', 'rb') as file:
						self.train_labels_shfl[shuffle_sz].append(pickle.load(file))

					if load_htmp:
						with open(simuset_path / 'shuffle_{:d}'.format(shuffle_sz) / 'labels_heatmap_shfl.pickle', 'rb') as file:
							self.lbl_htmp_shfl[shuffle_sz].append(pickle.load(file))

					if load_evals:
						self.eval_shfl[shuffle_sz].append(np.load(simuset_path / 'shuffle_{:d}'.format(shuffle_sz) / 'evaluation_shuffled.npy', allow_pickle=True))
					self.var_acc_shfl[shuffle_sz].append(np.load(simuset_path / 'shuffle_{:d}'.format(shuffle_sz) / 'var_shuffle_accuracy.npy'))
					self.var_pred_shfl[shuffle_sz].append(np.load(simuset_path / 'shuffle_{:d}'.format(shuffle_sz) / 'var_shuffle_classes_prediction.npy', allow_pickle=True))

			self.n_seqs[0] = len(self.train_labels_orig)
			for block_sz, train_labels in self.train_labels_shfl.items():
				self.n_seqs[block_sz] = len(train_labels)


			if load_atc:
				self.atc_orig.append(np.load(simuset_path /'autocorr_original.npy'))
				if load_shuffle:
					for shuffle_sz in self.shuffle_sizes:
						self.atc_shfl[shuffle_sz] = []
						self.atc_shfl[shuffle_sz].append(np.load(simuset_path / 'shuffle_{:d}'.format(shuffle_sz) / 'autocorr_shuffle.npy'))

		if load_atc:
			print("load_atc set to True. Autocorrelations loaded.")

		self.set_hsv(self.hue, self.uniform)


	def set_hsv(self, hue=0.5, uniform=False):
		self.hsv_orig = [0, 1, 0.9] if not uniform else [0, 0, 0.15]
		if not self.shuffle_sizes:
			return
		l_shfl = len(self.shuffle_sizes) if not uniform else 1
		sat_stride = 0.2/l_shfl
		value_stride = 0.8/l_shfl
		self.hsv_shfl_dict = {blck_sz: [hue, 1-sat_stride*shfl_id, 0.2+value_stride*shfl_id] for shfl_id, blck_sz in enumerate(self.shuffle_sizes)}


	def get_explr_times(self):
		"""
		Returns a list of the iteration IDs (one per sequence) of the first iteration at which all labels have been seen in the sequence at least once.
		If not all labels are explored, nothing is added to the list
		"""
		t_explrs = []

		for seq_id, seq in enumerate(self.train_labels_orig):
			t_explr = None
			obs_lbl_set = set()
			nobs_seq = []
			for itr_id, lbl in enumerate(seq):
				obs_lbl_set.add(lbl)
				if t_explr is None and len(obs_lbl_set) == self.n_classes:
					t_explr = itr_id
					t_explrs += [t_explr]
					break
		
		return t_explrs

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

		lbls_ax.plot(seq)

		obs_lbl_set = set()
		nobs_seq = []
		for itr_id, lbl in enumerate(seq):
			obs_lbl_set.add(lbl)
			nobs_seq.append(len(obs_lbl_set))
			if t_explr is None and len(obs_lbl_set) == self.n_classes:
				t_explr = itr_id

		lbls_ax.plot(nobs_seq)
		if strides is not None:
			for stride in strides:
				lbls_ax.axvline(x=stride, ymin=0, ymax=self.n_classes)

		ttl = 'History of labels in the original training sequence'
		if t_explr:
			ttl = ttl+' - tau_asym=' + str(t_explr)
		plt.title(ttl)

		return nobs_seq


	def lbl_distrib(self, max_iter:int=None, shuffled_blocksz=None, filter_perc:float=2, cumulative=False, save_formats=None, xaxis_n_ticks=10, multi_simus=False, figset_name=default_figset_name):
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

		distrib = np.zeros(shape=(self.seq_length, self.n_classes))

		fig, (heatmap_ax, distr_ax) = plt.subplots(nrows=2, ncols=1, figsize=(18,20))

		if shuffled_blocksz is None:
			n_seqs = self.n_seqs[0]
			if multi_simus:
				seq_set = self.train_labels_orig
			else:
				simu_id = random.randint(0,n_seqs-1)
				seq_set = [self.train_labels_orig[simu_id]]
		else:
			n_seqs = self.n_seqs[shuffled_blocksz]
			if multi_simus:
				seq_set = self.train_labels_shfl[shuffled_blocksz]
			else:
				simu_id = random.randint(0,n_seqs-1)
				seq_set = [self.train_labels_shfl[shuffled_blocksz][simu_id]]

		for seq in seq_set:
			if len(seq) > self.seq_length:
				seq = seq[0:self.seq_length]
			distrib[range(self.seq_length), seq] += 1/n_seqs

		# Optionally, cumulative summing over iterations
		if cumulative:
			norm = np.arange(1, 1+float(self.seq_length))
			distrib = np.divide(
				np.cumsum(distrib, axis=0).transpose(),
				norm
				).transpose()

		# Gaussian smoothing
		if max_iter is not None:
			filter_sz = filter_perc*max_iter/100
		else:
			filter_sz = filter_perc*self.seq_length/100
		filt_distrib = np.copy(distrib)
		for cl_id in range(self.n_classes):
			filt_distrib[:, cl_id] = gaussian_filter1d(distrib[:, cl_id], filter_sz)

		# Plot heatmap of different labels
		xtick_scale = self.n_tests//xaxis_n_ticks
		xtick_pos = int(self.seq_length/((self.n_tests//xtick_scale)))*np.arange((self.n_tests//xtick_scale)+1)

		if max_iter is not None:
			sns.heatmap(
				np.transpose(filt_distrib[0:max_iter,:]),
				ax = heatmap_ax,
				cbar=False
				)
		else:
			sns.heatmap(
				np.transpose(filt_distrib),
				ax = heatmap_ax,
				cbar=False
				)

		#heatmap_ax.set_xticks(xtick_pos)

		heatmap_ax.set_xlabel('Iteration')
		heatmap_ax.set_ylabel('Class label')

		# Plot all distributions on one plot
		for cl_id in range(self.n_classes):
			hsv = [0.8*cl_id/self.n_classes, 1, 0.7]
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
					figset_name,
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

				fig.savefig(out_filepath, format=fmt)

		return fig, heatmap_ax, distr_ax
			

#######################################################
### Functions to output accuracy=f(iteration) plots ###
#######################################################

def make_perfplot(rs, blocks, ax, plt_confinter=False, uniform=False, linewidth=3, draw_timescales=False, draw_explorations=False):
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

	x_labels = rs.var_acc_orig[0][:,1]


	### OPTIONNAL TIMESCALES ###
	if draw_timescales:
		single_timescales = sorted(list(set(rs.params["Timescales"])))
		for (timescale_id, timescale) in enumerate(single_timescales):
			if timescale > rs.seq_length:
				break
			ax.vlines(x=timescale, ymin=0, ymax=100, linewidth=3, color=(0.012,0.61,0.98), alpha=0.4, ls='--')
			ax.text(x=timescale+0.0125*rs.seq_length, y=50, s="{:.2E}".format(timescale), color=(0.012,0.61,0.98), fontsize=16, rotation=90, verticalalignment='center')

	### OPTIONNAL FULL EXPLORATION TIMES ###
	if draw_explorations:
		explr_times = rs.get_explr_times()
		for explr_time in explr_times:
			ax.vlines(x=explr_time, ymin=0, ymax=100, linewidth=2, color=(0.21,0.48,0.55), alpha=0.4, ls='--')
	### SHUFFLES ###
	if blocks is None and hasattr(rs, 'var_acc_shfl'):
		blocks = sorted(rs.blocks_sizes)

	for block_sz in blocks:
		if block_sz not in rs.var_acc_shfl.keys():
			continue
		n_shfl = len(rs.var_acc_shfl[block_sz])
		acc_data = rs.var_acc_shfl[block_sz]

		var_acc_shfl = np.mean([acc[:,0] for acc in acc_data], axis=0)
		var_acc_shfl_std = np.std([acc[:,0] for acc in acc_data], axis=0)
		ax.plot(
			x_labels, var_acc_shfl,
			ls = 'solid',
			linewidth = linewidth,
			color = hsv_to_rgb(rs.hsv_shfl_dict[block_sz]),
			label='S={0:d}'.format(block_sz)
		)

		if plt_confinter:
			ax.fill_between(
				x = x_labels,
				y1 = np.maximum(0, var_acc_shfl - conf_fscore[0.95]*np.sqrt(var_acc_shfl*(100-var_acc_shfl)/n_shfl)),
				y2 = np.minimum(var_acc_shfl + conf_fscore[0.95]*np.sqrt(var_acc_shfl*(100-var_acc_shfl)/n_shfl), 100),
				color = hsv_to_rgb(rs.hsv_shfl_dict[block_sz]),
				alpha = 0.2
			)

	### ORIGINAL ###
	n_orig = len(rs.var_acc_orig)
	var_acc_orig = np.mean([acc[:,0] for acc in rs.var_acc_orig], axis=0)
	var_acc_orig_std = np.std([acc[:, 0] for acc in rs.var_acc_orig], axis=0)
	ax.plot(
			x_labels, var_acc_orig,
			ls = 'solid' if not uniform else '--',
			linewidth = linewidth,
			color = hsv_to_rgb(rs.hsv_orig),
			label='No shuffle' if not uniform else 'Uniform sequence'
		)

	if plt_confinter:
		ax.fill_between(
			x = x_labels,
			y1 = np.maximum(0, var_acc_orig - conf_fscore[0.95]*np.sqrt(var_acc_orig*(100-var_acc_orig)/n_orig)),
			y2 = np.minimum(var_acc_orig + conf_fscore[0.95]*np.sqrt(var_acc_orig*(100-var_acc_orig)/n_orig), 100),
			color = hsv_to_rgb(rs.hsv_orig),
			alpha = 0.4
		)


def format_perf_plot(ax, title, legend_title, xtick_pos, xtick_labels, plot_window=None):
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
	#ax.set_position([box.x0, box.y0 + box.height * 0.2, box.width, box.height * 0.8])
	#ax.legend(fontsize=24, loc='upper center', bbox_to_anchor=(0.5, -0.1), title=legend_title, title_fontsize=22, frameon=True, fancybox=True, ncol=2)
	ax.legend(fontsize=24, loc='lower right', title=legend_title, title_fontsize=25, frameon=True, fancybox=True, ncol=2)

	ax.set_xlabel('Iterations', fontsize=24)
	ax.set_ylabel('Accuracy (%)', fontsize=24)

	ax.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
	#x_formatter = utils.OOMFormatter(5, "%1.1f")
	#ax.xaxis.set_major_formatter(x_formatter)

	#ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(8))

	if plot_window is not None:
		ax.set_xlim(plot_window[0], plot_window[1])


def make_perfplot_unit(
	rs, blocks, rs_unif=None,
	plot_window=None, blocks_to_plot='small',
	plt_confinter=False, n_ticks=10, save_formats=None, draw_timescales=False, draw_explorations=False, figset_name=default_figset_name
	):
	"""
	Creates accuracy plots from a single ResultSet object, comparing the result set vs baseline uniform, with shuffling block sizes in argument "blocks"

	Parameters
	----------
	rs : ResultSet
		main result set
	rs_unif : ResultSet
		baseline result set with uniform generation strategy
	n_tests: int
		number of tests during learning, corresponing to the number of time while moving forward on the sequence where we stop training and evaluate performance on a test set.
	plot_window: tuple like (x_min, x_max)
		if not None, will restrict plotting to a subset of iterations with (x_min, x_max)
	blocks: dict
		dictionnary containing, for a set of contexts (stored as keys) a list of block sizes to display, for the main ResultSet
	blocks_for_shared_plots: list(int)
		list of shuffle block sizes that were used for shuffling. Those will be plotted in dotted line in the third plot.
	plt_confinter: bool
		if True, confidence intervals for confidence level 95% will be printed.
		Defaults to False
		Note that because of the stochastic nature of sequence generation, the time of discovery of the different classes of the problem varies a lot, resulting in artificially exacerbated variance.
	n_ticks: int
		number of ticks to use on x-axis. Ticks will be uniformly distributed between 0 and rs.seq_length
	save_formats: list(str)
		if not None, plots will be saved in the provided formats
	"""
	xtick_scale = rs.n_tests//n_ticks
	xtick_pos = xtick_scale*np.arange((rs.n_tests//xtick_scale)+1)
	xtick_labels = int(rs.seq_length/((rs.n_tests//xtick_scale)))*np.arange((rs.n_tests//xtick_scale)+1)

	fig = plt.figure(figsize=(18,12))
	acc_ax = fig.add_subplot(1, 1, 1)

	### 1) UNIFORM + ORIGINAL ###
	if rs_unif is not None:
		make_perfplot(rs_unif, blocks=blocks[blocks_to_plot], ax=acc_ax, plt_confinter=plt_confinter, uniform=True)

	make_perfplot(rs, blocks=blocks[blocks_to_plot], ax=acc_ax, plt_confinter=plt_confinter, draw_timescales=draw_timescales, draw_explorations=draw_explorations)

	format_perf_plot(ax=acc_ax, title="Accuracy as a function of time for original and shuffled sequence - " + rs.descr, legend_title=rs.descr,
		xtick_pos=xtick_pos, xtick_labels=xtick_labels, plot_window=plot_window)


	if save_formats is not None:
		for fmt in save_formats:
			out_filepath = os.path.join(
				paths['plots'],
				figset_name,
				"accuracy/accuracy1_main{mainsetname:s}_unif{unifsetname:s}_{blocksz:s}blocks_{date:s}.{fmt:s}".format(
					mainsetname = rs.name,
					unifsetname = rs_unif.name,
					blocksz = blocks_to_plot,
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

			fig.savefig(out_filepath, format=fmt)

	return fig, acc_ax


def make_perfplot_comparison(
	rs, blocks, rs_altr=None, blocks_altr=None, rs_unif=None,
	plot_window=None, blocks_to_plot='small',
	plt_confinter=False, n_ticks=10, save_formats=None, draw_timescales=False, draw_explorations=False, figset_name=default_figset_name
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
	n_tests: int
		number of tests during learning, corresponing to the number of time while moving forward on the sequence where we stop training and evaluate performance on a test set.
	plot_window: tuple like (x_min, x_max)
		if not None, will restrict plotting to a subset of iterations with (x_min, x_max)
	blocks: dict
		dictionnary containing, for a set of contexts (stored as keys) a list of block sizes to display, for the main ResultSet
	blocks_altr: dict
		dictionnary containing, for a set of contexts (stored as keys) a list of block sizes to display, for the alternative ResultSet.
		If none is provided and an alternative set is displayed, the block dictionnary provided for the main set will be used
	blocks_for_shared_plots: list(int)
		list of shuffle block sizes that were used for shuffling. Those will be plotted in dotted line in the third plot.
	plt_confinter: bool
		if True, confidence intervals for confidence level 95% will be printed.
		Defaults to False
		Note that because of the stochastic nature of sequence generation, the time of discovery of the different classes of the problem varies a lot, resulting in artificially exacerbated variance.
	n_ticks: int
		number of ticks to use on x-axis. Ticks will be uniformly distributed between 0 and rs.seq_length
	save_formats: list(str)
		if not None, plots will be saved in the provided formats
	"""
	xtick_scale = rs.n_tests//n_ticks
	xtick_pos = xtick_scale*np.arange((rs.n_tests//xtick_scale)+1)
	seq_length = max(rs.seq_length, rs_altr.seq_length, rs_unif.seq_length)

	xtick_labels = int(seq_length/((rs.n_tests//xtick_scale)))*np.arange((rs.n_tests//xtick_scale)+1)

	n_plots = 1 + 2*int(rs_altr is not None)
	fig = plt.figure(figsize=(18*n_plots,12))
	axes = []

	acc_ax = fig.add_subplot(1, n_plots, 1)
	axes.append(acc_ax)

	### 1) UNIFORM + ORIGINAL ###
	if rs_unif is not None:
		make_perfplot(rs_unif, blocks=blocks[blocks_to_plot], ax=acc_ax, plt_confinter=plt_confinter, uniform=True)

	make_perfplot(rs, blocks=blocks[blocks_to_plot], ax=acc_ax, plt_confinter=plt_confinter, draw_timescales=draw_timescales, draw_explorations=draw_explorations)

	format_perf_plot(ax=acc_ax, title="Accuracy as a function of time for original and shuffled sequence - " + rs.descr, legend_title=rs.descr,
		xtick_pos=xtick_pos, xtick_labels=xtick_labels, plot_window=plot_window)


	### 2) UNIFORM + ALTERNATIVE ###
	if rs_altr is not None:
		acc_ax_altr = fig.add_subplot(1, n_plots, 2)
		axes.append(acc_ax_altr)

		if blocks_altr is None:
			blocks_altr = blocks

		if rs_unif is not None:
			make_perfplot(rs_unif, blocks=blocks[blocks_to_plot], ax=acc_ax_altr, plt_confinter=plt_confinter, uniform=True)

		make_perfplot(rs_altr, blocks=blocks_altr[blocks_to_plot], ax=acc_ax_altr, plt_confinter=plt_confinter, draw_timescales=draw_timescales, draw_explorations=draw_explorations)

		format_perf_plot(ax=acc_ax_altr, title="Accuracy as a function of time for original and shuffled sequence - " + rs.descr, legend_title=rs.descr,
			xtick_pos=xtick_pos, xtick_labels=xtick_labels, plot_window=plot_window)


	### 3) ALL SCENARIOS, REDUCED NUMBER OF BLOCKS ###
	if rs_altr is not None:
		acc_ax_all = fig.add_subplot(1, n_plots, 3)
		axes.append(acc_ax_all)

		if rs_unif is not None:
			make_perfplot(rs_unif, blocks=blocks['acc_plots_shared'], ax=acc_ax_all, plt_confinter=plt_confinter, uniform=True)

		make_perfplot(rs_altr, blocks=blocks_altr['acc_plots_shared'], ax=acc_ax_all, plt_confinter=plt_confinter, draw_timescales=draw_timescales, draw_explorations=draw_explorations)
		make_perfplot(rs, blocks=blocks['acc_plots_shared'], ax=acc_ax_all, plt_confinter=plt_confinter)

		format_perf_plot(ax=acc_ax_all, title="Comparative accuracy as a function of time for different scenarios", legend_title=rs.descr,
			xtick_pos=xtick_pos, xtick_labels=xtick_labels, plot_window=plot_window)

	fig.tight_layout(pad=10.0)


	if save_formats is not None:
		for fmt in save_formats:
			out_filepath = os.path.join(
				paths['plots'],
				figset_name,
				"accuracy/accuracy3_main{mainsetname:s}_altr{altersetname:s}_unif{unifsetname:s}_{blocksz:s}blocks_{date:s}.{fmt:s}".format(
					mainsetname = rs.name,
					altersetname = rs_altr.name,
					unifsetname = rs_unif.name,
					blocksz = blocks_to_plot,
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

			fig.savefig(out_filepath, format=fmt)

	return fig, axes

def make_perfplot_matrix(
	rs_list, blocks, rs_unif=None,
	plot_window=None, blocks_to_plot='small',
	plt_confinter=False, n_ticks=10, save_formats=None, draw_timescales=False, draw_explorations=False, figset_name=default_figset_name
	):
	"""
	Creates a matrix of accuracy plots for a list of ResultSets, with:
	- diagonal elements being the plots of classification performance as a function of time for original and shuffled sequences
	- non-diagonal elements comparing two ResultSet classification performances

	Parameters
	----------
	rs_list : list(ResultSet)
		list of ResultSet objects that will be used for plotting on the matrix
		Warning: all ResultSet in the list should be run of sequences of the same length
	rs_unif : ResultSet
		baseline result set with uniform generation strategy
	n_tests: int
		number of tests during learning, corresponing to the number of time while moving forward on the sequence where we stop training and evaluate performance on a test set.
	plot_window: tuple like (x_min, x_max)
		if not None, will restrict plotting to a subset of iterations with (x_min, x_max)
	blocks: dict
		dictionnary containing, for a set of contexts (stored as keys) a list of block sizes to display, for the main ResultSet
	blocks_altr: dict
		dictionnary containing, for a set of contexts (stored as keys) a list of block sizes to display, for the alternative ResultSet.
		If none is provided and an alternative set is displayed, the block dictionnary provided for the main set will be used
	blocks_for_shared_plots: list(int)
		list of shuffle block sizes that were used for shuffling. Those will be plotted in dotted line in the third plot.
	plt_confinter: bool
		if True, confidence intervals for confidence level 95% will be printed.
		Defaults to False
		Note that because of the stochastic nature of sequence generation, the time of discovery of the different classes of the problem varies a lot, resulting in artificially exacerbated variance.
	n_ticks: int
		number of ticks to use on x-axis. Ticks will be uniformly distributed between 0 and rs_list[0].seq_length
	save_formats: list(str)
		if not None, plots will be saved in the provided formats
	"""
	import itertools

	n_rs = len(rs_list)
	fig, axes = plt.subplots(n_rs, n_rs, sharex=True, sharey=True, figsize=(15*n_rs,15*n_rs))

	### Diagonal elements
	for rs_id, rs in enumerate(rs_list):
		if rs_unif is not None:
			make_perfplot(rs_unif, blocks=blocks[blocks_to_plot], ax=axes[rs_id, rs_id], plt_confinter=plt_confinter, uniform=True)

		make_perfplot(rs, blocks=blocks[blocks_to_plot], ax=axes[rs_id, rs_id], plt_confinter=plt_confinter, draw_timescales=draw_timescales, draw_explorations=draw_explorations)

		axes[rs_id, rs_id].set_facecolor((0.87, 0.87, 0.87))
		axes[rs_id, rs_id].legend()
		if plot_window is not None:
			axes[rs_id, rs_id].set_xlim(plot_window[0], plot_window[1])


	### Non-diagonal elements
	for rs1_id, rs2_id in itertools.permutations(range(n_rs), 2):
		if rs_unif is not None:
			make_perfplot(rs_unif, blocks=blocks[blocks_to_plot], ax=axes[rs1_id, rs2_id], plt_confinter=plt_confinter, uniform=True)

		rs1 = rs_list[rs1_id]
		rs2 = rs_list[rs2_id]
		make_perfplot(rs1, blocks=blocks[blocks_to_plot], ax=axes[rs1_id, rs2_id], plt_confinter=plt_confinter, draw_timescales=draw_timescales, draw_explorations=draw_explorations)
		make_perfplot(rs2, blocks=blocks[blocks_to_plot], ax=axes[rs1_id, rs2_id], plt_confinter=plt_confinter)

		axes[rs1_id, rs2_id].legend()
		if plot_window is not None:
			axes[rs1_id, rs2_id].set_xlim(plot_window[0], plot_window[1])

	fig.tight_layout(pad=10.0)


	if save_formats is not None:
		for fmt in save_formats:
			out_filepath = os.path.join(
				paths['plots'],
				figset_name,
				"accuracy/accuracyMatrix_{rs_name_:s}_{blocksz:s}blocks_{date:s}.{fmt:s}".format(
					rs_name_ = rs_list[0].name,
					blocksz = blocks_to_plot,
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

			fig.savefig(out_filepath, format=fmt)

	return fig, axes


######################################################
### Catastrophic Forgetting Score graphs functions ###
######################################################

def get_cf(lbl_seq, acc_orig, acc_shfl1, n_labels, conv_crit=0.005, plot=False):
	"""
	Computes the PLTS score based on a sequence, the accuracy curve of the original sequence and the accuracy score of the fully shuffled (block size 1) sequence

	Parameters
	----------
	lbl_seq : list(int)
		sequence of labels for this simulation. This is used to determine the timeframe of exploration of the different classes.
	acc_orig : list(float)
		list of accuracy scores (%) for the original sequence
	acc_shfl1 : list(float)
		list of accuracy scores (%) for the sequence shuffled with a block size of 1, meaning that on learning, the algorithm is only limited by the incomplete exploration of classes as well as the imbalance in the sequence up to this point.
		The imbalance is the same as in the original sequence at any point in time. 
	plot: bool
		if True, will plot the RPLF as a function of the number of iterations since the first full exploration

	Returns
	-------
	cf: list
		list of the PLTS scores from the time of first full exploration of the classes.
		Therefore the scores are aligned from the first iteration that all classes were explored, onward.
		If at least one class is missing from the sequence, then None
	n_tests_before_fullexplr: int
		The number of classification accuracy tests that occured before all classes were seen at least once in the sequence.
		If at least one class is missing from the sequence, then None
	"""
	n_tests = len(acc_orig)
	seq_length = len(lbl_seq)
	res = seq_length//n_tests
	n_tests_before_fullexplr = None
	n_tests_to_convergence = None

	obs_lbl_set = set()
	nobs_seq = []
	for itr_id, lbl in enumerate(lbl_seq):
		obs_lbl_set.add(lbl)
		nobs_seq.append(len(obs_lbl_set))
		if len(obs_lbl_set) == n_labels:
			n_tests_before_fullexplr = itr_id//res
			break # TO REMOVE FOR DEFINITION (2) OF CF

	## TO ENABLE FOR DEFINITION (2) OF CF
	#spl_nobs_seq = [nobs_seq[k*res] for k in range(n_tests)]
	#try:
	#	assert(len(spl_nobs_seq) == len(acc_orig) == len(acc_shfl1))
	#except:
	#	pdb.set_trace()

	for test_id in range(n_tests):
		if (acc_orig[test_id] > 100*(1-conv_crit)) and (acc_shfl1[test_id] > 100*(1-conv_crit)):
			n_tests_to_convergence = test_id
			break

	accarr_shfl1 = np.array(acc_shfl1)
	accarr_orig = np.array(acc_orig)
	#auc_shfl1 = 0.5*res*(accarr_shfl1 + np.roll(accarr_shfl1, 1))
	auc_shfl1 = 0.5*(accarr_shfl1 + np.roll(accarr_shfl1, 1)) #AUC computed on the previous slice
	auc_shfl1[0] = 0
	#auc_orig = 0.5*res*(accarr_orig + np.roll(accarr_orig, 1))
	auc_orig = 0.5*(accarr_orig + np.roll(accarr_orig, 1))
	auc_orig[0] = 0

	## CORRESPONDING DEFINITION OF CF (1): ABSOLUTE LOSS IS CLASSIFICATION SCORE AT ANY POINT IN TIME
	cf = auc_shfl1 - auc_orig

	## CORRESPONDING DEFINITION OF CF (2): %LOSS IS ACHIEVABLE CLASSIFICATION SCORE AT ANY POINT IN TIME 
	#cf = (np.array(acc_shfl1)-np.array(acc_orig))/np.array(spl_nobs_seq)

	if plot:
		fig = plt.figure(1, figsize=(18,12))
		cf_ax = plt.subplot(111)
		cf_ax.plot(
			cf,
			label='Forgetting score as a fct of #iteration'
		)

	return np.concatenate((cf[2:], np.array([0, 0])), axis=0), n_tests_before_fullexplr, n_tests_to_convergence


def get_cf_history(rs, blocks,
	xtick_scale=25, plt_confinter=False, save_formats=None, confidence=0.05, cf_correctionfactor=None, figset_name=default_figset_name):

	tot_cf = {}
	tot_cf_std = {}
	tot_cf_ci = {}
	avg_cf = {}
	avg_cf_std = {}
	avg_cf_ci = {}

	tot_ald_cf = {}
	tot_ald_cf_std = {}
	tot_ald_cf_ci = {}
	avg_ald_cf = {}
	avg_ald_cf_std = {}
	avg_ald_cf_ci = {}

	if 'MNIST' in rs.name:
		conv_crit=0.015
	else:
		conv_crit=0.005

	xtick_pos = xtick_scale*np.arange((rs.n_tests//xtick_scale)+1)
	xtick_labels = int(rs.seq_length/((rs.n_tests//xtick_scale)))*np.arange((rs.n_tests//xtick_scale)+1)

	fig_cfscore = plt.figure(figsize=(18,28))
	cf_ax = plt.subplot(2,1,1)
	cf_ald_ax = plt.subplot(2,1,2)

	t_explr_list = []
	cf_orig_list = []
	cf_orig_aligned_list = []
	cf_shfl_lists = {block_sz: [] for block_sz in rs.var_acc_shfl.keys()}
	cf_shfl_aligned_lists = {block_sz: [] for block_sz in rs.var_acc_shfl.keys()}

	for seq_id, seq in enumerate(rs.train_labels_orig):

		orig_cf, n_tests_to_fullexplr, n_tests_to_convergence = get_cf( # Returns a SET of AUC integral history, at the same resolution as the accuracy curve	
			seq,
			rs.var_acc_orig[seq_id][:,0],
			rs.var_acc_shfl[1][seq_id][:,0],
			rs.n_classes,
			conv_crit=conv_crit,
		)

		if n_tests_to_fullexplr is not None:
			t_explr_list.append(n_tests_to_fullexplr)

			cf_aligned = np.concatenate([
				np.array(orig_cf[n_tests_to_fullexplr:]),
				np.zeros(n_tests_to_fullexplr)
			])
			cf_orig_list.append(orig_cf)
			cf_orig_aligned_list.append(cf_aligned)

			shfl_cf = {}				
			for block_sz in rs.var_acc_shfl.keys():
				assert block_sz in rs.block_sizes
				shfl_cf[block_sz], _, _ = get_cf(
					seq,
					rs.var_acc_shfl[block_sz][seq_id][:,0],
					rs.var_acc_shfl[1][seq_id][:,0],
					rs.n_classes,
					conv_crit=conv_crit
				)

				cf_aligned = np.concatenate([
					np.array(shfl_cf[block_sz][n_tests_to_fullexplr:]),
					np.zeros(n_tests_to_fullexplr)
				])
				cf_shfl_lists[block_sz].append(shfl_cf[block_sz])
				cf_shfl_aligned_lists[block_sz].append(cf_aligned)

	cf_mean = np.mean( # this provides a history curve (CF as a function of time from the first time of full exploration), computed as the mean of the simulations where full exploration occured
		np.stack(cf_orig_list, axis=1),
		axis=1
	)
	cf_std = np.std(
		np.stack(cf_orig_list, axis=1),
		axis=1
	)
	if cf_correctionfactor is not None:
		cf_mean *= cf_correctionfactor
		cf_std *= cf_correctionfactor

	tot_cf[0] = rs.seq_length*np.mean(cf_mean) # This is where we turn an area between curves history into a TOTAL area between curves
	tot_cf_std[0] = rs.seq_length*np.mean(cf_std)
	tot_cf_ci[0] = scipy.stats.t.ppf(q=1-0.5*confidence,df=rs.n_seqs[0])*tot_cf_std[0]

	#pdb.set_trace()
	avg_cf[0] = (rs.seq_length/(n_tests_to_convergence*rs.res_temp))*np.mean(cf_mean) # This is where we turn an area between curves history into an AVERAGE area between curves, computed FROM START OF SEQUENCE TO CONVERGENCE POINT
	avg_cf_std[0] = (rs.seq_length/(n_tests_to_convergence*rs.res_temp))*np.mean(cf_std)
	avg_cf_ci[0] = scipy.stats.t.ppf(q=1-0.5*confidence,df=rs.n_seqs[0])*avg_cf_std[0]

	cf_ax.plot(
		cf_mean,
		color = hsv_to_rgb(rs.hsv_orig),
		ls = 'solid',
		label = rs.descr + ' - Original sequence'
	)

	if plt_confinter:
		cf_ax.fill_between(
			x = range(len(cf_mean)),
			y1 = cf_mean - scipy.stats.t.ppf(q=1-0.5*confidence,df=rs.n_seqs[0])*cf_std,
			y2 = cf_mean + scipy.stats.t.ppf(q=1-0.5*confidence,df=rs.n_seqs[0])*cf_std,
			color = hsv_to_rgb(rs.hsv_orig),
			alpha = 0.4
		)

	for block_sz in rs.var_acc_shfl.keys():
		cf_mean = np.mean( # this provides a history curve (CF as a function of time from the first time of full exploration), computed as the mean of the simulations where full exploration occured
			np.stack(cf_shfl_lists[block_sz], axis=1),
			axis=1
		)
		cf_std = np.std(
			np.stack(cf_shfl_lists[block_sz], axis=1),
			axis=1
		)
		if cf_correctionfactor is not None:
			cf_mean *= cf_correctionfactor
			cf_std *= cf_correctionfactor

		tot_cf[block_sz] = rs.seq_length*np.mean(cf_mean) # This is where we turn an area between curves history into a TOTAL area between curves
		tot_cf_std[block_sz] = rs.seq_length*np.mean(cf_std)
		tot_cf_ci[block_sz] = scipy.stats.t.ppf(q=1-0.5*confidence,df=rs.n_seqs[block_sz])*tot_cf_std[block_sz]

		avg_cf[block_sz] = (rs.seq_length/(n_tests_to_convergence*rs.res_temp))*np.mean(cf_mean)
		avg_cf_std[block_sz] = (rs.seq_length/(n_tests_to_convergence*rs.res_temp))*np.mean(cf_std)
		avg_cf_ci[block_sz] = scipy.stats.t.ppf(q=1-0.5*confidence,df=rs.n_seqs[block_sz])*avg_cf_std[block_sz]

		if block_sz in blocks['cfhist_plots']:
			cf_ax.plot(
				cf_mean,
				color = hsv_to_rgb(rs.hsv_shfl_dict[block_sz]),
				ls = 'solid',
				label = rs.descr + ' - Shuffled w/ block size {0:d}'.format(block_sz)
			)

			if plt_confinter:
				cf_ax.fill_between(
					x = range(len(cf_mean)),
					y1 = cf_mean - scipy.stats.t.ppf(q=1-0.5*confidence,df=rs.n_seqs[block_sz])*cf_std,
					y2 = cf_mean + scipy.stats.t.ppf(q=1-0.5*confidence,df=rs.n_seqs[block_sz])*cf_std,
					color = hsv_to_rgb(rs.hsv_shfl_dict[block_sz]),
					alpha = 0.4
				)

	if len(cf_orig_aligned_list) > 0:
		cf_aligned_mean = np.mean( # this provides a history curve (CF as a function of time from the first time of full exploration), computed as the mean of the simulations where full exploration occured
			np.stack(cf_orig_aligned_list, axis=1),
			axis=1
		)
		cf_aligned_std = np.std(
			np.stack(cf_orig_aligned_list, axis=1),
			axis=1
		)
		if cf_correctionfactor is not None:
			cf_aligned_mean *= cf_correctionfactor
			cf_aligned_std *= cf_correctionfactor

		tot_ald_cf[0] = rs.seq_length*np.mean(cf_aligned_mean)
		tot_ald_cf_std[0] = rs.seq_length*np.mean(cf_aligned_std)
		tot_ald_cf_ci[0] = scipy.stats.t.ppf(q=1-0.5*confidence,df=rs.n_seqs[0])*tot_ald_cf_std[0]

		avg_ald_cf[0] = (rs.seq_length/((n_tests_to_convergence-n_tests_to_fullexplr)*rs.res_temp))*np.mean(cf_aligned_mean)
		avg_ald_cf_std[0] = (rs.seq_length/((n_tests_to_convergence-n_tests_to_fullexplr)*rs.res_temp))*np.mean(cf_aligned_std)
		avg_ald_cf_ci[0] = scipy.stats.t.ppf(q=1-0.5*confidence,df=rs.n_seqs[0])*avg_ald_cf_std[0]

		cf_ald_ax.plot(
			cf_aligned_mean,
			color = hsv_to_rgb(rs.hsv_orig),
			ls = 'solid',
			label = rs.descr + ' - Original sequence'
		)

		if plt_confinter:
			cf_ald_ax.fill_between(
				x = range(len(cf_aligned_mean)),
				y1 = cf_aligned_mean - scipy.stats.t.ppf(q=1-0.5*confidence,df=rs.n_seqs[0])*cf_aligned_std,
				y2 = cf_aligned_mean + scipy.stats.t.ppf(q=1-0.5*confidence,df=rs.n_seqs[0])*cf_aligned_std,
				color = hsv_to_rgb(rs.hsv_orig),
				alpha = 0.4
			)

		for block_sz in rs.var_acc_shfl.keys():
			cf_aligned_mean = np.mean( # this provides a history curve (CF as a function of time from the first time of full exploration), computed as the mean of the simulations where full exploration occured
				np.stack(cf_shfl_aligned_lists[block_sz], axis=1),
				axis=1
			)
			cf_aligned_std = np.std(
				np.stack(cf_shfl_aligned_lists[block_sz], axis=1),
				axis=1
			)
			if cf_correctionfactor is not None:
				cf_aligned_mean *= cf_correctionfactor
				cf_aligned_std *= cf_correctionfactor

			tot_ald_cf[block_sz] = rs.seq_length*np.mean(cf_aligned_mean)
			tot_ald_cf_std[block_sz] = rs.seq_length*np.mean(cf_aligned_std)
			tot_ald_cf_ci[block_sz] = scipy.stats.t.ppf(q=1-0.5*confidence,df=rs.n_seqs[block_sz])*tot_ald_cf_std[block_sz]

			avg_ald_cf[block_sz] = (rs.seq_length/((n_tests_to_convergence-n_tests_to_fullexplr)*rs.res_temp))*np.mean(cf_aligned_mean)
			avg_ald_cf_std[block_sz] = (rs.seq_length/((n_tests_to_convergence-n_tests_to_fullexplr)*rs.res_temp))*np.mean(cf_aligned_std)
			avg_ald_cf_ci[block_sz] = scipy.stats.t.ppf(q=1-0.5*confidence,df=rs.n_seqs[block_sz])*avg_ald_cf_std[block_sz]

			if block_sz in blocks['cfhist_plots']:
				cf_ald_ax.plot(
					cf_aligned_mean,
					color = hsv_to_rgb(rs.hsv_shfl_dict[block_sz]),
					ls = 'solid',
					label = rs.descr + ' - Shuffled w/ block size {0:d}'.format(block_sz)
				)

				if plt_confinter:
					cf_ald_ax.fill_between(
						x = range(len(cf_aligned_mean)),
						y1 = cf_aligned_mean - scipy.stats.t.ppf(q=1-0.5*confidence,df=rs.n_seqs[block_sz])*cf_aligned_std,
						y2 = cf_aligned_mean + scipy.stats.t.ppf(q=1-0.5*confidence,df=rs.n_seqs[block_sz])*cf_aligned_std,
						color = hsv_to_rgb(rs.hsv_shfl_dict[block_sz]),
						alpha = 0.4
					)


	box = cf_ax.get_position()
	cf_ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])

	lgd = cf_ax.legend(loc='upper right', ncol=1, prop={'size': 16})

	xlabels = cf_ax.set_xlabel('Iterations', fontsize=14)
	ylabels = cf_ax.set_ylabel('Accuracy loss from CF (%)', fontsize=14)

	box = cf_ald_ax.get_position()
	cf_ald_ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])

	lgd = cf_ald_ax.legend(loc='upper right', ncol=1, prop={'size': 16})

	xlabels = cf_ald_ax.set_xlabel('Iterations', fontsize=14)
	ylabels = cf_ald_ax.set_ylabel('Accuracy loss from CF (%)', fontsize=14)

	fig_cfscore.tight_layout(pad=10.0)
	if save_formats is not None:
		for fmt in save_formats:
			out_filepath = os.path.join(
				paths['plots'],
				figset_name,
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

			fig_cfscore.savefig(out_filepath, format=fmt, bbox_extra_artists=(lgd,), bbox_inches='tight')


	return tot_cf, tot_cf_ci, avg_cf, avg_cf_ci, tot_ald_cf, tot_ald_cf_ci, avg_ald_cf, avg_ald_cf_ci


def plot_cf_profile(cf_stats, alignment_method="raw", metric="mean", rs_names=None, x_origpos=3e5, vline_pos=1e5, xlog=False, ylog=False, save_formats=None, cfprof_ymax=None, normalize=False, plot_timescales=False, figset_name=default_figset_name):
	"""
	Produces plots of the CF score as a function of 
	"""
	fig_mean_cfs = plt.figure(figsize=(18,12))
	ax_mean_cfs = plt.subplot(111)

	min_y = None
	max_y = None

	for rs_name, cf_set in cf_stats.items():
		rs = cf_set['rs']
		_cf_data = cf_set['data']
		_cf_models = cf_set['model_fit']
		cf_model_fit = None

		if alignment_method == "raw":
			if metric == "mean":
				cf_data = _cf_data['avg_raw_cf']
				cf_ci = _cf_data['avg_raw_cf_ci']
				if 'avg_raw_cf' in _cf_models.keys():
					cf_model_fit = _cf_models['avg_raw_cf']
			elif metric == "total":
				cf_data = _cf_data['tot_raw_cf']
				cf_ci = _cf_data['tot_raw_cf_ci']
				if 'tot_raw_cf' in _cf_models.keys():
					cf_model_fit = _cf_models['tot_raw_cf']
		elif alignment_method == "aligned":
			if metric == "mean":
				cf_data = _cf_data['avg_ald_cf']
				cf_ci = _cf_data['avg_ald_cf_ci']
				if 'avg_ald_cf' in _cf_models.keys():
					cf_model_fit = _cf_models['avg_ald_cf']
			elif metric == "total":
				cf_data = _cf_data['tot_ald_cf']
				cf_ci = _cf_data['tot_ald_cf_ci']
				if 'tot_ald_cf' in _cf_models.keys():
					cf_model_fit = _cf_models['tot_ald_cf']

		xtick_pos = [k for k in sorted(cf_data.keys()) if k>0]

		if normalize and cf_data[0]>0:
			cf_data = {k: v/cf_data[0] for k,v in cf_data.items()}
			#if cf_model_fit is not None:
			#	TO IMPLEMENT: MODEL FIT NORMALIZATION HANDLING

		# Showing timescales in the background with vertical lines
		if plot_timescales:
			timescales = rs.params["Timescales"]
			for ts in timescales:
				ax_mean_cfs.vlines(x=ts, ymin=0, ymax=1.1*x_origpos, linestyles='--', linewidth=2, color=[0.42,0.74,0.95])

		sorted_block_sizes = [block_sz for block_sz in sorted(cf_data.keys()) if block_sz>0]

		# Plotting fitted models, if any have been computed
		if cf_model_fit is not None:
			res_mult = 1.25
			x_range_mult = sorted_block_sizes[-1]/sorted_block_sizes[0]
			x_range = [sorted_block_sizes[0]*(res_mult**k) for k in range(math.ceil(math.log10(x_range_mult)/math.log10(res_mult))+1)]
			# print(cf_model_fit['params'])
			# print(x_range)
			# print([cf_model_fit['function'](x, *cf_model_fit['params']) for x in x_range])
			# pdb.set_trace()
			ax_mean_cfs.plot(
				x_range,
				[cf_model_fit['function'](x, *cf_model_fit['params']) for x in x_range],
				ls = 'solid',
				linewidth=3,
				color = hsv_to_rgb(rs.hsv_shfl_dict[rs.shuffle_sizes[-1]]),
				label = rs.descr
			)

		# Plotting data points

		ax_mean_cfs.plot(
			xtick_pos,
			[cf_data[k] for k in sorted_block_sizes],
			ls = 'solid' if cf_model_fit is None else 'none',
			linewidth=3,
			marker = 'o',
			markersize = 15,
			markeredgewidth = 3,
			color = hsv_to_rgb(rs.hsv_shfl_dict[rs.shuffle_sizes[-1]]),
			label = rs.descr
		)
		ax_mean_cfs.plot(
			xtick_pos,
			[cf_data[k] for k in sorted_block_sizes],
			ls = 'none',
			marker = 'o',
			markersize = 10,
			markeredgewidth = 3,
			color = "white"
		)
		ax_mean_cfs.fill_between(
			x = xtick_pos,
			y1 = [cf_data[k] - cf_ci[k] for k in sorted_block_sizes],
			y2 = [cf_data[k] + cf_ci[k] for k in sorted_block_sizes],
			color = hsv_to_rgb(rs.hsv_shfl_dict[rs.shuffle_sizes[-1]]),
			alpha = 0.08
		)

		ax_mean_cfs.plot(
			x_origpos,
			cf_data[0] - cf_ci[0],
			marker = '_',
			markersize = 10,
			markeredgewidth = 4,
			color = hsv_to_rgb(rs.hsv_shfl_dict[rs.shuffle_sizes[-1]])
		)
		ax_mean_cfs.plot(
			x_origpos,
			cf_data[0] + cf_ci[0],
			marker = '_',
			markersize = 10,
			markeredgewidth = 4,
			color = hsv_to_rgb(rs.hsv_shfl_dict[rs.shuffle_sizes[-1]])
		)
		ax_mean_cfs.plot(
			[x_origpos, x_origpos],
			[cf_data[0] - cf_ci[0], cf_data[0] + cf_ci[0]],
			color = hsv_to_rgb(rs.hsv_shfl_dict[rs.shuffle_sizes[-1]]),
			alpha = 0.08
		)

		ax_mean_cfs.plot(
			x_origpos,
			cf_data[0],
			marker = 'o',
			markersize = 20,
			markeredgewidth = 4,
			color = hsv_to_rgb(rs.hsv_shfl_dict[rs.shuffle_sizes[-1]])
		)
		ax_mean_cfs.plot(
			x_origpos,
			cf_data[0],
			marker = 'o',
			markersize = 12,
			markeredgewidth = 4,
			color = "white"
		)

		#plt.xticks(xtick_pos)
		ax_mean_cfs.hlines(y=cf_data[0], xmin=0, xmax=1.1*x_origpos, linestyles=':', linewidth=3, color = hsv_to_rgb(rs.hsv_orig))

		if min_y is None or (np.min(list(cf_data.values())) < min_y):
			min_y = np.min(list(cf_data.values()))
		if max_y is None or (np.max(list(cf_data.values())) > max_y):
			max_y = np.max(list(cf_data.values()))

	##############################
	# Plot formatting for paper ##
	##############################

	#ax_mean_cfs.set_title('Per-label loss in classification performance as a function of shuffle block size', fontsize = 18)

	lgd = ax_mean_cfs.legend(fancybox=True, shadow=True, prop={'size': 24}, loc='upper left')

	ax_mean_cfs.set_xlabel('Shuffle length', fontsize=24)
	if metric == "mean":
		ax_mean_cfs.set_ylabel('Average classification score loss from CF (%)', fontsize=24)
	elif metric == "total":
		ax_mean_cfs.set_ylabel('Total classification score loss from CF (% * iter)', fontsize=24)

	fig_mean_cfs.tight_layout(pad=10.0)


	if xlog:
		ax_mean_cfs.set_xscale("log")

	if ylog: 
		ax_mean_cfs.set_yscale("log")

	#for tick in ax_mean_cfs.xaxis.get_major_ticks():
	#	tick.label.set_rotation('vertical')

	max_blocksz = max(list(cf_data.keys()))
	ax_mean_cfs.set_xlim(0, 1.1 * x_origpos)

	if cfprof_ymax is None:
		ax_mean_cfs.set_ylim(min_y, 1.15*max_y)
	else:
		ax_mean_cfs.set_ylim(min_y, cfprof_ymax)

	ax_mean_cfs.vlines(x=np.sqrt(x_origpos*max_blocksz), ymin=min_y, ymax=1.15*max_y, linewidth=3, color="black")
	#ax_mean_cfs.set_ylim(-5, 12)

	# Saving figure
	if save_formats is not None:
		for fmt in save_formats:
			out_filepath = os.path.join(
				paths['plots'],
				figset_name,
				"CFscore/profile/{date:s}/{alignment_method:s}_{metric:s}/PLTS_profile_x{xscl:s}_y{yscl:s}_{norm:s}.{fmt:s}".format(
					alignment_method = alignment_method,
					metric = metric,
					xscl = "log" if xlog is True else "lin",
					yscl = "log" if ylog is True else "lin",
					norm = "_norm" if normalize else "",
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

			fig_mean_cfs.savefig(out_filepath, format=fmt)

	return fig_mean_cfs, ax_mean_cfs


def fit_profile(cf_stat, version='sigmoid'):
	from scipy.optimize import curve_fit

	def cut_sigmoid(x, L, x0, k, xsat):
		y = L / (1 + np.exp(-k*(np.minimum(np.log10(x), np.log10(xsat))-np.log10(x0))))
		return (y)

	def sigmoid(x, L, x0, k):
	    y = L / (1 + np.exp(-k*(np.log10(x)-np.log10(x0))))
	    return (y)

	def sigmoid_with_cte(x, L, x0, k, b):
	    y = L / (1 + np.exp(-k*(np.log10(x)-np.log10(x0))))+b
	    return (y)

	def sigmoid_primitive(x, L, x0, k):
	    y = L*((np.log10(x)-np.log10(x0))+(1/k)*np.log(1+np.exp(-k*(np.log10(x)-np.log10(x0)))))
	    return (y)

	def sigmoid_of_log(x, L, x0, k):
		y = L / (1 + np.power(x0/x,k))
		return (y)

	cf_stat['model_fit'] = {}

	for stat_type in ['avg_raw_cf', 'tot_raw_cf', 'avg_ald_cf', 'tot_ald_cf']:
		x_data = np.fromiter(cf_stat['data'][stat_type].keys(), dtype=int)
		y_data = np.fromiter(cf_stat['data'][stat_type].values(), dtype=float)
		norm_y_data = y_data/y_data[0]

		if version == 'sigmoid':
			p0 = [max(y_data), np.median(x_data), 1] # this is an mandatory initial guess
			norm_p0 = [1, np.median(x_data), 1]
			model = sigmoid
		elif version == 'sigmoid_primitive':
			p0 = [(y_data[-1] - y_data[-4])/(x_data[-1] - x_data[-4]), np.median(x_data), 1]
			norm_p0 = [(y_data[-1] - y_data[-4])/(x_data[-1] - x_data[-4]), np.median(x_data), 1]
			model = sigmoid_primitive
		elif version == 'cut_sigmoid':
			y_diff = y_data-np.roll(y_data,1)
			y_diff[0] = 0
			x_sat_est = x_data[np.argmax(y_diff)]
			p0 = [max(y_data), np.median(x_data), 1, x_sat_est] # this is an mandatory initial guess
			norm_p0 = [1, np.median(x_data), 1, min(y_data), x_sat_est]
			model = cut_sigmoid

		popt, pcov = curve_fit(model, x_data, y_data, p0, method='lm')
		norm_popt, norm_pcov = curve_fit(model, x_data, norm_y_data, norm_p0, method='lm')

		if version=='sigmoid':
			max_steepness = 1/popt[2] #(popt[0]*popt[2])/(4*popt[1]*np.log(10)) for actual maximum steepness from PLTS profile fit
			norm_max_steepness = 1/norm_popt[2] #(norm_popt[0]*norm_popt[2])/(4*norm_popt[1]*np.log(10)) #(norm_popt[0]*norm_popt[2])/4
		elif version=='sigmoid_primitive':
			max_steepness = popt[0]
			norm_max_steepness = norm_popt[0]
		else:
			max_steepness = 1/popt[2] #(popt[0]*popt[2])/(4*popt[1]*np.log(10))
			norm_max_steepness = 1/norm_popt[2] #(norm_popt[0]*norm_popt[2])/(4*norm_popt[1]*np.log(10))

		cf_stat['model_fit'][stat_type] = {}

		cf_stat['model_fit'][stat_type]['params'] = popt
		cf_stat['model_fit'][stat_type]['params_cov'] = pcov
		cf_stat['model_fit'][stat_type]['function'] = lambda x, *popt: model(x, *popt)
		cf_stat['model_fit'][stat_type]['max_steepness'] = max_steepness

		cf_stat['model_fit'][stat_type]['norm_params'] = norm_popt
		cf_stat['model_fit'][stat_type]['norm_params_cov'] = norm_pcov
		cf_stat['model_fit'][stat_type]['norm_function'] = lambda x, *popt: model(x, *norm_popt)
		cf_stat['model_fit'][stat_type]['norm_max_steepness'] = norm_max_steepness


def report_steepness(cf_stats, depths, metric="avg", bf=20, normalize=True, confidence=0.05, ylog_steepness=False, ylog_maxval=False, save_formats=None, figset_name=default_figset_name):

	def get_top_barplot_value(barplots):
		top_height = 0

		for barplot in barplots:
			for bar in barplot:
				top_height = np.maximum(top_height,bar.get_height())

		return top_height

	width = 0.15
	metric_sepr = 0.08
	x_ind = np.arange(len(depths))
	plts_steepness = {}
	plts_maxval = {}
	ticks = []
	ticklabels = []

	if normalize:
		steepness_data = 'norm_max_steepness'
		params_cov = 'norm_params_cov'
		ylbl = r"Inverses of steepness of sigmoid fitted to normalized $PL_{TS}$ data"
	else:
		steepness_data = 'max_steepness'
		params_cov = 'params_cov'
		ylbl = r"Inverse of steepness of sigmoid fitted to $PL_{TS}$ data"

	steepness_bars = []
	maxval_bars = []

	hsv = {"Ultra": [0.6, 1, 0.9], "Rb": [0.3, 1, 0.9]}

	fig_report = plt.figure(figsize=(7*len(depths),24))
	steepness_ax = plt.subplot(2,1,1)
	maxval_ax = plt.subplot(2,1,2)

	for alignment_method_id, alignment_method in enumerate(["raw", "ald"]):
		stat_name = "{0:s}_{1:s}_cf".format(metric, alignment_method)
		stat_ci_name = "{0:s}_{1:s}_cf_ci".format(metric, alignment_method)
		plts_steepness[stat_name] = {}
		plts_steepness[stat_ci_name] = {}
		plts_maxval[stat_name] = {}
		plts_maxval[stat_ci_name] = {}
		for seq_type_id, seq_type in enumerate(["Ultra", "Rb"]):
			plts_steepness[stat_name][seq_type] = []
			plts_steepness[stat_ci_name][seq_type] = []
			plts_maxval[stat_name][seq_type] = []
			plts_maxval[stat_ci_name][seq_type] = []
			for depth_id, depth in enumerate(depths):
				rs_name = "artificial_d{depth_:d}{seq_type_:s}Mixed{bf_:d}bits".format(depth_ = depth, seq_type_ = seq_type, bf_ = bf)
				pdb.set_trace()
				rs = cf_stats[rs_name]["rs"]

				plts_steepness[stat_name][seq_type].append(cf_stats[rs_name]['model_fit'][stat_name][steepness_data])
				plts_maxval[stat_name][seq_type].append(cf_stats[rs_name]['data'][stat_name][0])

				steepness_var = cf_stats[rs_name]['model_fit'][stat_name][params_cov][2]
				steepness_ci = scipy.stats.t.ppf(q=1-0.5*confidence,df=rs.n_seqs[0])*np.sqrt(steepness_var)

				plts_steepness[stat_ci_name][seq_type].append(steepness_ci)
				plts_maxval[stat_ci_name][seq_type].append(cf_stats[rs_name]['data'][stat_ci_name][0])

				ticks.append(x_ind[depth_id] + 2*width*(alignment_method_id-1) + metric_sepr*(alignment_method_id-0.5) + width*seq_type_id )
				ticklabels.append("{alignment_method_:s}\n{seq_type_:s}".format(alignment_method_=alignment_method, seq_type_=seq_type))

			steepness_bars.append(
				steepness_ax.bar(
					x_ind + 2*width*(alignment_method_id-1) + metric_sepr*(alignment_method_id-0.5) + width*seq_type_id,
					plts_steepness[stat_name][seq_type],
					width,
					yerr = None, #yerr = plts_steepness[stat_ci_name][seq_type],
					label = "{seq_type_:s}_{stat_name_:s}".format(seq_type_=seq_type, stat_name_=stat_name),
					color = hsv_to_rgb(hsv[seq_type])
				)
			)

			maxval_bars.append(
				maxval_ax.bar(
					x_ind + 2*width*(alignment_method_id-1) + metric_sepr*(alignment_method_id-0.5) + width*seq_type_id,
					plts_maxval[stat_name][seq_type],
					width,
					yerr = None, #yerr = plts_steepness[stat_ci_name][seq_type],
					label = "{seq_type_:s}_{stat_name_:s}".format(seq_type_=seq_type, stat_name_=stat_name),
					color = hsv_to_rgb(hsv[seq_type])
				)
			)

	top_steepness = get_top_barplot_value(steepness_bars)
	top_maxval = get_top_barplot_value(maxval_bars)

	steepness_ax.set_xticks(ticks)
	steepness_ax.tick_params(axis=u'x', which=u'both',length=0)
	steepness_ax.set_xticklabels(ticklabels)

	maxval_ax.set_xticks(ticks)
	maxval_ax.tick_params(axis=u'x', which=u'both',length=0)
	maxval_ax.set_xticklabels(ticklabels)

	for depth_id, depth in enumerate(depths):
		steepness_ax.annotate(
			"Depth {depth_:d}".format(depth_=depth),
			xy = (x_ind[depth_id] - 0.5*width, top_steepness),
			xytext = (0, 7),
			textcoords="offset points",
			fontsize = 28,
			ha='center', va='bottom'
		)

		maxval_ax.annotate(
			"Depth {depth_:d}".format(depth_=depth),
			xy = (x_ind[depth_id] - 0.5*width, top_maxval),
			xytext = (0, 7),
			textcoords="offset points",
			fontsize = 28,
			ha='center', va='bottom'
		)

	if ylog_steepness:
		steepness_ax.set_yscale("log")
	if ylog_maxval:
		maxval_ax.set_yscale("log")

	steepness_ax.set_ylabel(ylbl, fontsize=24)
	maxval_ax.set_ylabel(r"$PL_{TS}$ of original sequence (%)", fontsize=24)

	# Saving figure
	if save_formats is not None:
		for fmt in save_formats:
			out_filepath = os.path.join(
				paths['plots'],
				figset_name,
				"CFscore/steepness/{date:s}/PLTS_steepness_{metric_:s}{norm_:s}.{fmt:s}".format(
					metric_ = metric,
					date = datetime.datetime.now().strftime("%Y%m%d"),
					norm_ = "_norm" if normalize else "",
					fmt = fmt
				)
			)

			if not os.path.exists(os.path.dirname(out_filepath)):
			    try:
			        os.makedirs(os.path.dirname(out_filepath))
			    except OSError as exc: # Guard against race condition
			        if exc.errno != errno.EEXIST:
			            raise

			fig_report.savefig(out_filepath, format=fmt)
	