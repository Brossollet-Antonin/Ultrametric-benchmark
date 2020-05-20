from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt
from matplotlib.colors import hsv_to_rgb
from matplotlib import rcParams

import os, sys, ast, pdb
import pickle
import numpy as np
import pandas as pd

import argparse
import utils
paths = utils.get_project_paths()

import result_loader as ld
#ld.format_paper()

## PARAMETERS COMMON TO ALL SIMULATIONS ##
seq_length = 300000
n_tests = 300
T = 0.4
n_batches = 10
lr=0.01
artificial_seq_len = 200

d5_sm_splt_sizes = (1, 200, 500, 1000)
d5_lg_splt_sizes = (1000, 2000, 8000, 20000, 40000, 80000)
d5_all_splt_sizes = (1, 200, 500, 1000, 2000, 8000, 20000, 40000, 80000)

d7_sm_splt_sizes = (1, 82, 164, 328)
d7_lg_splt_sizes = (328, 1312, 5248, 20992, 41984, 83968)
d7_all_splt_sizes = (1, 328, 1312, 20992, 83968)

## ARGUMENT PARSING
parser = argparse.ArgumentParser('./make_figures.py', description='Produces figures')
parser.add_argument('--bf_ratio', type=float, help="Bit-flipping ratio per level in the ultrametric tree that generate patterns")
parser.add_argument('--make_lbl_history', type=int, default=0, help="Whether to output label history heatmaps and curve-like figures (takes a little time)")


def load_rs_artificial(bf_ratio, sim_directory, artificial_seq_len=200, seq_length=300000, n_tests=300, T=0.4, dataset='artificial_32', nnarchi='FCL20'):
	bit_flips_per_lvl = int(bf_ratio*artificial_seq_len)

	rs = {}
	### ULTRAMETRIC GENERATION METHOD, NO SHUFFLING CLASSES
	rs['um_um'] = ld.ResultSet(
		rs_name="d5UltraUnmixed{:d}bits".format(bit_flips_per_lvl),
		rs_descr="Ultrametric depth 5 (unmixed labels, {0:d} bits/lvl)".format(bit_flips_per_lvl),
		sim_map_dict = sim_directory,
		dataset_name = dataset,
		nn_config = nnarchi,
		seq_type = 'ultrametric_ratio{:d}_noclshfl'.format(bit_flips_per_lvl),
		simset_id = T,
		hue = 0
	)
	rs['um_um'].load_analytics()


	### ULTRAMETRIC GENERATION METHOD, SHUFFLING CLASSES
	## With 8bits changed at every level down in the ultrametric tree for pattern generation
	rs['um_mx'] = ld.ResultSet(
		rs_name="d5UltraMixed{:d}bits".format(bit_flips_per_lvl),
		rs_descr="Ultrametric depth 5 (mixed labels, {:d} bits/lvl)".format(bit_flips_per_lvl),
		sim_map_dict = sim_directory,
		dataset_name = dataset,
		nn_config = nnarchi,
		seq_type = 'ultrametric_ratio{:d}'.format(bit_flips_per_lvl),
		simset_id = T,
		hue = 0.12
	)
	rs['um_mx'].load_analytics()


	### RANDOM BLOCKS (PAIRS OF LABELS) GENERATION METHOD
	## With 8bits changed at every level down in the ultrametric tree for pattern generation
	rs['blck_mx'] = ld.ResultSet(
		rs_name="d5RbMixed{:d}bits".format(bit_flips_per_lvl),
		rs_descr="Random blocks (paired mixed labels, {:d} bits/lvl), 32 classes".format(bit_flips_per_lvl),
		sim_map_dict = sim_directory,
		dataset_name = dataset,
		nn_config = nnarchi,
		seq_type = 'random_blocks2_ratio{:d}'.format(bit_flips_per_lvl),
		simset_id = 1000,
		hue = 0.5
	)
	rs['blck_mx'].load_analytics()


	### RANDOM BLOCKS (PAIRS OF LABELS) GENERATION METHOD
	## With 8bits changed at every level down in the ultrametric tree for pattern generation
	rs['blck_um'] = ld.ResultSet(
		rs_name="d5RbUnmixed{:d}bits".format(bit_flips_per_lvl),
		rs_descr="Random blocks (paired unmixed labels, {:d} bits/lvl), 32 classes".format(bit_flips_per_lvl),
		sim_map_dict = sim_directory,
		dataset_name = dataset,
		nn_config = nnarchi,
		seq_type = 'random_blocks2_ratio{:d}_noclshfl'.format(bit_flips_per_lvl),
		simset_id = 1000,
		hue = 0.62
	)
	rs['blck_um'].load_analytics()


	### UNIFORM PROBABILITY DISTIRUBTION
	rs['unif'] = ld.ResultSet(
		rs_name="d5Unif{:d}bits".format(bit_flips_per_lvl),
		rs_descr="Uniform, 32 classes",
		sim_map_dict = sim_directory,
		dataset_name = dataset,
		nn_config = nnarchi,
		seq_type = 'uniform_ratio{:d}'.format(bit_flips_per_lvl),
		simset_id = 0.0,
		hue = 0
	)
	rs['unif'].load_analytics(load_shuffle=False)

	return rs


def make_CFfigures_artificial(bf_ratio, sim_directory, all_split_sz, sm_split_sz, lg_split_sz, save_formats=['svg', 'pdf'], cfprof_method='mean', cfprof_x_origpos=8.5e4, cfprof_var_scale=0.5, cfprof_ylog=True, artificial_seq_len=200, seq_length=300000, n_tests=300, T=0.4, dataset='artificial_32', nnarchi='FCL20'):

	## LOADING RESULT SETS
	rss = load_rs_artificial(bf_ratio, sim_directory, artificial_seq_len=artificial_seq_len, seq_length=seq_length, n_tests=n_tests, T=T, dataset=dataset, nnarchi=nnarchi)

	## Producing accuracy plots
	to_compare = [
		('um_mx', 'blck_mx'),
		('um_um', 'blck_um'),
		('um_mx', 'um_um'),
		('blck_mx', 'blck_um')
	]

	for (main_set, altr_set) in to_compare:
		ld.get_acc(
			rs=rss[main_set], rs_altr=rss[altr_set], rs_unif=rss['unif'],
			seq_length=seq_length, n_tests=n_tests, blocks=sm_split_sz, blocks_sizes = 'small',
			save_formats=save_formats
		)
		ld.get_acc(
			rs=rss[main_set], rs_altr=rss[altr_set], rs_unif=rss['unif'],
			seq_length=seq_length, n_tests=n_tests, blocks=lg_split_sz, blocks_sizes = 'large',
			save_formats=save_formats
		)

	# Making CF history plots
	cf_stats = {
		'name': "allCFplots_{:d}flips".format(int(bf_ratio*artificial_seq_len)),
		'stat_list': []
	}
	for rs_id, rs in rss.items():
		if (rs_id == 'unif'):
			continue
		_cf_stats = {}

		avg_cf, avg_cf_std, init_cf, init_cf_std = ld.load_cf_set(rs, all_split_sz, save_formats=save_formats)
		_cf_stats['rs'] = rs
		_cf_stats['rs_name'] = rs.name
		_cf_stats['avg_cf'] = avg_cf
		_cf_stats['avg_cf_std'] = avg_cf_std
		_cf_stats['init_cf'] = init_cf
		_cf_stats['init_cf_std'] = init_cf_std

		cf_stats['stat_list'].append(_cf_stats)

	# Making CD profile plots
	fig_cfs, ax_cfs = ld.plot_cf_profile(cf_stats, method=cfprof_method, x_origpos=cfprof_x_origpos, var_scale=cfprof_var_scale, xlog=False, ylog=cfprof_ylog, save_formats=save_formats)
	#ld.plot_cf_profile(cf_stats, method=cfprof_method, x_origpos=cfprof_x_origpos, var_scale=cfprof_var_scale, xlog=True, ylog=cfprof_ylog, save_formats=save_formats)

	# Release memory
	del rss
	return cf_stats, fig_cfs, ax_cfs


def make_lblhistory_artificial(sim_directory, first_iters_focus=None, save_formats=['svg', 'pdf'], artificial_seq_len=200, seq_length=300000, n_tests=300, T=0.4, dataset='artificial_32', nnarchi='FCL20'):
	## LOADING RESULT SETS
	rss = load_rs_artificial(0.1, sim_directory, artificial_seq_len=artificial_seq_len, seq_length=seq_length, n_tests=n_tests, T=T, dataset=dataset, nnarchi=nnarchi)

	if first_iters_focus is not None:
		print("Ultrametric, instantaneous, first {:d} iters only...".format(first_iters_focus))
		rss['um_mx'].lbl_distrib(max_iter=first_iters_focus, save_formats=save_formats)
		print("Done")
		
		print("Ultrametric, cumulative, first {:d} iters only...".format(first_iters_focus))
		rss['um_mx'].lbl_distrib(max_iter=first_iters_focus, cumulative=True, save_formats=save_formats)
		print("Done")

		print("Random blocks, instantaneous, first {:d} iters only...".format(first_iters_focus))
		rss['blck_mx'].lbl_distrib(max_iter=first_iters_focus, save_formats=save_formats)
		print("Done")
		
		print("Random blocks, cumulative, first {:d} iters only...".format(first_iters_focus))
		rss['blck_mx'].lbl_distrib(max_iter=first_iters_focus, cumulative=True, save_formats=save_formats)
		print("Done")

	else:
		print("Ultrametric, instantaneous, whole sequence")
		rss['um_mx'].lbl_distrib(save_formats=save_formats)
		print("Done")
		
		print("Ultrametric, cumulative, whole sequence")
		rss['um_mx'].lbl_distrib(cumulative=True, save_formats=save_formats)
		print("Done")

		print("Random blocks, instantaneous, whole sequence")
		rss['blck_mx'].lbl_distrib(save_formats=save_formats)
		print("Done")
		
		print("Random blocks, cumulative, whole sequence")
		rss['blck_mx'].lbl_distrib(cumulative=True, save_formats=save_formats)
		print("Done")

	del rss


if __name__ == '__main__':
	args = parser.parse_args()

	## LOADING MAPPER FILE ##
	with open(paths['simus']+'simu_mapping_compact.txt', 'r', encoding='utf-8') as filenames:
		filenames_dct_txt = filenames.read().replace('\n', '')
		
	sim_directory = ast.literal_eval(filenames_dct_txt)
	plt.ion()

	print("Making CF-related figures...")
	fig_cfs, ax_cfs = make_CFfigures_artificial(args.bf_ratio, sim_directory, d5_all_splt_sizes, d5_sm_splt_sizes, d5_lg_splt_sizes, save_formats=['svg', 'pdf'], cfprof_method='mean', cfprof_x_origpos=8.5e4, cfprof_var_scale=0.5, cfprof_ylog=True, artificial_seq_len=200, seq_length=300000, n_tests=300, T=0.4, dataset='artificial_32', nnarchi='FCL20')
	print("Done")

	if args.make_lbl_history:
		print("Making label history figures...")
		make_lblhistory_artificial(sim_directory, first_iters_focus=50000, save_formats=['svg', 'pdf'], artificial_seq_len=200, seq_length=300000, n_tests=300, T=0.4, dataset='artificial_32', nnarchi='FCL20')
		print("Done")

		print("Making label history figures...")
		make_lblhistory_artificial(sim_directory, save_formats=['svg', 'pdf'], artificial_seq_len=200, seq_length=300000, n_tests=300, T=0.4, dataset='artificial_32', nnarchi='FCL20')
		print("Done")