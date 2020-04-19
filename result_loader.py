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

hsv_um_um_orig = (0, 1, 1)
hsv_um_um_shfl_list = tuple([0, 1-block_id*0.08, 1] for block_id in range(8))

hsv_um_mx_orig = (0.32, 0.9, 0.65)
hsv_um_mx_shfl_list = tuple([0.32, 0.9-block_id*0.08, 0.65] for block_id in range(8))

hsv_rb_um_orig = (0.63, 1, 0.8)
hsv_rb_um_shfl_list = tuple([0.63, 1-block_id*0.08, 0.8] for block_id in range(8))

hsv_rb_mx_orig = (0.77, 1, 0.8)
hsv_rb_mx_shfl_list = tuple([0.77, 1-block_id*0.08, 0.8] for block_id in range(8))

markers = ['o','+','x','4','s','p','P', '8', 'h', 'X']

conf_fscore={
	0.9: 1.64,
	0.95: 1.96,
	0.98: 2.33,
	0.99: 2.58,
}


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
	def __init__(self, set_name, sim_map_dict, dataroot, sim_struct, dataset_name, nn_config, seq_type, simset_id, hsv_orig, hsv_shfl_list=None):
		self.set_name = set_name
		self.sim_map_dict = sim_map_dict
		self.dataroot = dataroot
		self.sim_struct = sim_struct
		self.dataset_name = dataset_name
		self.nn_config = nn_config
		self.seq_type = seq_type
		self.simset_id = simset_id
		self.hsv_orig = hsv_orig
		self.hsv_shfl_list = hsv_shfl_list


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

		if self.seq_type in ("uniform", "ultrametric", "ultrametric_noclshfl"):
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

		if load_data:
			self.train_data_orig = []
			self.train_data_shfl = []

		if load_atc:
			self.atc_orig = []
			self.atc_shfl = []

		for simuset_path in os.listdir(folderpath):
			
			if 'T'+str(T) not in simuset_path:
				continue 

			os.chdir(folderpath+'/'+simuset_path)

			with open('train_labels_orig.pickle', 'rb') as file:
				self.train_labels_orig.append(pickle.load(file))

			with open('distribution_train.pickle', 'rb') as file:
				self.dstr_train.append(pickle.load(file))
				
			with open('parameters.json', 'r') as file:
				self.params.append(json.load(file))

			self.eval_orig.append(np.load('evaluation_original.npy', allow_pickle=True))
			self.var_acc_orig.append(np.load('var_original_accuracy.npy'))
			self.var_pred_orig.append(np.load('var_original_classes_prediction.npy', allow_pickle=True))

			if load_htmp:
				with open('labels_heatmap_shfl.pickle', 'rb') as file:
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

					with open('shuffle_'+str(shuffle_sz)+'/train_labels_shfl.pickle', 'rb') as file:
						self.train_labels_shfl[shuffle_sz].append(pickle.load(file))

					if load_htmp:
						with open('shuffle_'+str(shuffle_sz)+'/labels_heatmap_shfl.pickle', 'rb') as file:
							self.lbl_htmp_shfl[shuffle_sz].append(pickle.load(file))

					self.eval_shfl[shuffle_sz].append(np.load('shuffle_'+str(shuffle_sz)+'/evaluation_shuffled.npy', allow_pickle=True))
					self.var_acc_shfl[shuffle_sz].append(np.load('shuffle_'+str(shuffle_sz)+'/var_shuffle_accuracy.npy'))
					self.var_pred_shfl[shuffle_sz].append(np.load('shuffle_'+str(shuffle_sz)+'/var_shuffle_classes_prediction.npy', allow_pickle=True))

			if load_data:
				print("Loading data for {0:s}...".format(datapath))

				with open('train_data_orig.pickle', 'rb') as file:
					self.train_data_orig.append(pickle.load(file))

				if load_shuffle:
					for shuffle_sz in self.shuffle_sizes:
						self.train_data_shfl[shuffle_sz] = []
						with open('shuffle_'+str(shuffle_sz)+'/train_data_shfl.pickle', 'rb') as file:
							self.train_data_shfl[shuffle_sz].append(pickle.load(file))
					
				print("...done")

			if load_atc:
				self.atc_orig.append(np.load('autocorr_original.npy'))
				if load_shuffle:
					for shuffle_sz in self.shuffle_sizes:
						self.atc_shfl[shuffle_sz] = []
						self.atc_shfl[shuffle_sz].append(np.load('shuffle_'+str(shuffle_sz)+'/autocorr_shuffle.npy'))

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
				occur_id = random.randint(0, len(self.train_labels_orig)-1)
				seq = self.train_labels_orig[occur_id]
			else:
				occur_id = random.randint(0, len(self.train_labels_shfl[shuffled_blocksz])-1)
				seq = self.train_labels_shfl[shuffled_blocksz][occur_id]

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


def make_perfplot(rs, blocks, ax, plt_confinter=False):
	### ORIGINAL ###
	n_orig = len(rs.var_acc_orig)
	var_acc_orig = np.mean([acc[:,0] for acc in rs.var_acc_orig], axis=0)
	var_acc_orig_std = np.std([acc[:,0] for acc in rs.var_acc_orig], axis=0)
	ax.plot(
			var_acc_orig,
			ls = 'solid',
			color = hsv_to_rgb(rs.hsv_orig),
			label='Ultrametric sequence - No shuffling'
		)

	if plt_confinter:
		ax.fill_between(
			x = range(len(var_acc_orig)),
			y1 = np.maximum(0, var_acc_orig - conf_fscore[0.95]*np.sqrt(var_acc_orig*(100-var_acc_orig)/n_orig)),
			y2 = np.minimum(var_acc_orig + conf_fscore[0.95]*np.sqrt(var_acc_orig*(100-var_acc_orig)/n_orig), 100),
			color = hsv_to_rgb(hsv_orig),
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
			var_acc_shfl,
			ls = '--',
			color = hsv_to_rgb(rs.hsv_shfl_list[block_id]),
			label='Ultrametric sequence - Shuffled w/ block size {0:d}'.format(block_sz)
		)

		if plt_confinter:
			ax.fill_between(
				x = range(len(var_acc_shfl)),
				y1 = np.maximum(0, var_acc_shfl - conf_fscore[0.95]*np.sqrt(var_acc_shfl*(100-var_acc_shfl)/n_shfl)),
				y2 = np.minimum(var_acc_shfl + conf_fscore[0.95]*np.sqrt(var_acc_shfl*(100-var_acc_shfl)/n_shfl), 100),
				color = hsv_to_rgb(rs.hsv_shfl_list[block_id]),
				alpha = 0.2
			)


def format_perf_plot(ax, title, xtick_pos, xtick_labels, plot_window=None):
	ax.set_xticks(xtick_pos)
	ax.set_xticklabels(xtick_labels)
	ax.set_title(title, fontsize = 14)

	box = ax.get_position()
	ax.set_position([box.x0, box.y0 + box.height * 0.1,
					 box.width, box.height * 0.9])

	ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
				  fancybox=True, shadow=True, ncol=2,
				  prop={'size': 16})

	ax.set_xlabel('Iterations', fontsize=14)
	ax.set_ylabel('Accuracy (%)', fontsize=14)

	if plot_window is not None:
		ax.set_xlim(plot_window[0], plot_window[1])

	for tick in ax.xaxis.get_major_ticks():
		tick.label.set_fontsize(14)
	
	for tick in ax.yaxis.get_major_ticks():
		tick.label.set_fontsize(14)


def get_acc(
	rs, rs_altr=None, rs_unif=None,
	seq_length=300000, n_tests=300, plot_window=None, blocks=None,
	blocks_for_shared_plots=None, plt_confinter=False, n_ticks=10, save_format=None
	):

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
	
	format_perf_plot(acc_ax, "Accuracy as a function of time for original and shuffled sequence - " + rs.set_name, xtick_pos, xtick_labels, plot_window)


	### 2) UNIFORM + ALTERNATIVE ###
	if rs_altr is not None:
		acc_ax_altr = plt.subplot(n_plots, 1, 2)
		axes.append(acc_ax_altr)

		if rs_unif is not None:
			make_perfplot(rs_unif, blocks=blocks, ax=acc_ax_altr, plt_confinter=plt_confinter)
		
		make_perfplot(rs_altr, blocks=blocks, ax=acc_ax_altr, plt_confinter=plt_confinter)
		
		format_perf_plot(acc_ax_altr, "Accuracy as a function of time for original and shuffled sequence - " + rs_altr.set_name, xtick_pos, xtick_labels, plot_window)


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

	if save_format is not None:
		plt.savefig('out_plots_acc.'+str(save_format), format=save_format)

	return fig, axes




def get_raw_cf(lbl_seq, acc_orig, acc_unif, plot=False):
	nspl = len(acc_orig)
	seql = len(lbl_seq)
	t_explr = None
	n_labels = len(set(lbl_seq))

	obs_lbl_set = set()
	nobs_seq = []
	for itr_id, lbl in enumerate(lbl_seq):
		obs_lbl_set.add(lbl)
		nobs_seq.append(len(obs_lbl_set))
		if t_explr is None and len(obs_lbl_set) == n_labels:
			t_explr = itr_id//(seql//nspl)

	spl_nobs_seq = [nobs_seq[k*(seql//nspl)] for k in range(nspl)]
	assert(len(spl_nobs_seq) == len(acc_orig) == len(acc_unif))

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

	for seq_id, seq in enumerate(rs.train_labels_orig):
		t_explr = []
		cf = []
		_cf, _t_explr = get_raw_cf(seq, rs.var_acc_orig[seq_id][:,0], rs.var_acc_shfl[1][seq_id][:,0])
		if _t_explr:
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
			label = rs.set_name + ' - Original sequence'
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
				rs.var_acc_shfl[1][seq_id][:,0]
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
				label = rs.set_name + ' - Shuffled w/ block size {0:d}'.format(block_sz)
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
	xtick_scale=25, plt_confinter=False, save_format=None
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
	if save_format is not None:
		plt.savefig('out_plots_cfscore.svg', format=save_format)

	return avg_cf, avg_cf_std, init_cf, init_cf_std


def plot_cf_profile(cf_sets, method='mean', x_origpos=2.5e4, vline_pos=2.2e4, xlog=False, ylog=False, var_scale=1):
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
		    label = rs.set_name
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
		ax_mean_cfs.fill_between(
		    x = [x_origpos],
		    y1 = [cf[0] - var_scale*cf_std[0]],
		    y2 = [cf[0] + var_scale*cf_std[0]],
		    color = hsv_to_rgb(rs.hsv_orig),
		    alpha = 0.2
		)

		ax_mean_cfs.hlines(y=cf[0], xmin=0, xmax=1.1*x_origpos, linestyles=':', linewidth=3, color = hsv_to_rgb(rs.hsv_orig))
		ax_mean_cfs.vlines(x=vline_pos, ymin=-0.1, ymax=1)

	# Plot formatting for figure 4 of paper

	#xtick_pos = [k for k in xtick_pos] + [x_origpos]
	#xtick_labels = [str(k) for k in xtick_pos] + [25000]
	ax_mean_cfs.set_xticks(xtick_pos)
	ax_mean_cfs.set_xticklabels(xtick_labels)

	ax_mean_cfs.set_title('Per-label loss in classification performance as a function of shuffle block size', fontsize = 18)

	ax_mean_cfs.legend(fancybox=True, shadow=True, prop={'size': 16})

	ax_mean_cfs.set_xlabel('Iterations', fontsize=16)
	ax_mean_cfs.set_ylabel('Average per-label loss from CF (%)', fontsize=16)

	fig_mean_cfs.tight_layout(pad=10.0)


	if xlog:
		ax_mean_cfs.set_xscale("log")
	if ylog:
		ax_mean_cfs.set_yscale("log")

	for tick in ax_mean_cfs.xaxis.get_major_ticks():
	    tick.label.set_fontsize(14) 
	    tick.label.set_rotation('vertical')
	    
	for tick in ax_mean_cfs.yaxis.get_major_ticks():
	    tick.label.set_fontsize(14) 

	ax_mean_cfs.set_xlim(0, 1.1*x_origpos)
	ax_mean_cfs.set_ylim(-0.1, 0.8)

	# Saving figure

	plt.savefig('out_plots_cfscore_avg_linscale.svg', format='svg')
	plt.savefig('out_plots_cfscore_avg_linscale.pdf', format='pdf')