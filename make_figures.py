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
from result_set import RS_DIR
#ld.format_paper()

## PARAMETERS COMMON TO ALL SIMULATIONS ##
seq_length = 300000
n_tests = 300
T = 0.4
n_batches = 10
lr=0.01
artificial_seq_len = 200

blocks = {
	5: {
		'all': (1, 200, 500, 1000, 2000, 8000, 20000, 40000, 80000),
		'small': (1, 200, 500, 1000),
		'large': (1000, 2000, 8000, 20000, 40000, 80000),
		'acc_plots_shared': (1, 500, 1000, 2000, 20000, 80000),
		'cfhist_plots': (1, 500, 1000, 2000, 20000, 80000)
	},
	7: {
		'all': (1, 82, 164, 328, 1312, 5248, 20992, 41984, 83968),
		'small': (1, 82, 164, 328),
		'large': (328, 1312, 5248, 20992, 41984, 83968),
		'acc_plots_shared': (1, 328, 1312, 20992, 83968),
		'cfhist_plots': (1, 328, 1312, 20992, 83968)
	}
}

bf_ratio_tested = (0.04, 0.07, 0.1, 0.13)
tree_depths_tested = (5, 7)


## ARGUMENT PARSING
parser = argparse.ArgumentParser('./make_figures.py', description='Produces figures')
parser.add_argument('--result_battery', type=float, choices=["ultra_vs_rb2", "compare_bit_flipping_mixed", "compare_bit_flipping_unmixed", "influence_of_tree_depth_mixed", "influence_of_tree_depth_unmixed"], help="Battery of results to generate graphs for")
parser.add_argument('--bf_ratio', type=float, help="Bit-flipping ratio per level in the ultrametric tree that generate patterns")
parser.add_argument('--make_lbl_history', type=int, default=0, help="Whether to output label history heatmaps and curve-like figures (takes a little time)")
parser.add_argument('--bf_ratio', default=0.1)
parser.add_argument('--tree_depth', default=5)

parser.add_argument('--first_iters_focus', type=int, default=50000)


class FigureSet:
	def __init__(self, rs_names, accuracy_to_compare, rs_for_lbl_plots, seq_length=300000, n_tests=300, artificial_seq_len=200):
		self.seq_length = seq_length
		self.n_tests = n_tests
		self.artificial_seq_len = artificial_seq_len
		self.rs_names = rs_names
		self.accuracy_to_compare = accuracy_to_compare
		self.rs_for_lbl_plots = rs_for_lbl_plots

	def load_result_sets(self):
		self.rs = {}
		for rs_name, rs_hue in self.rs_names.items():
			self.rs[rs_name] = RS_DIR[rs_name]
			self.rs[rs_name].load_analytics(hue=rs_hue)


def make_CFfigures_artificial(fs, blocks, save_formats=['svg', 'pdf'], cfprof_method='mean', cfprof_x_origpos=8.5e4, cfprof_var_scale=0.5, cfprof_ylog=False, wipe_mem=True):

	## Producing accuracy plots
	print("Making accuracy figures...")
	for (main_set, altr_set, unif_set) in fs.accuracy_to_compare:
		print("    {:s} vs {:s}".format(main_set, altr_set))
		ld.get_acc(
			rs=fs.rs[main_set], blocks=blocks, rs_altr=fs.rs[altr_set], rs_unif=fs.rs[unif_set],
			seq_length=fs.seq_length, n_tests=fs.n_tests,
			blocks_to_plot = 'small', save_formats=save_formats
		)
		ld.get_acc(
			rs=fs.rs[main_set], blocks=blocks, rs_altr=fs.rs[altr_set], rs_unif=fs.rs[unif_set],
			seq_length=fs.seq_length, n_tests=fs.n_tests,
			blocks_to_plot = 'large', save_formats=save_formats
		)

	# Making CF history plots
	fs.cf_stats = {
		'name': fs.name,
		'stat_list': []
	}

	print("Computing CF histories")
	for rs_name, rs in fs.rs.items():
		print("    {:s}".format(rs_name))
		if ("Unif" in rs_name):
			continue
		_cf_stats = {}

		avg_cf, avg_cf_std, init_cf, init_cf_std = ld.load_cf_set(rs, blocks, save_formats=save_formats)
		_cf_stats['rs'] = rs
		_cf_stats['rs_name'] = rs.name
		_cf_stats['avg_cf'] = avg_cf
		_cf_stats['avg_cf_std'] = avg_cf_std
		_cf_stats['init_cf'] = init_cf
		_cf_stats['init_cf_std'] = init_cf_std

		cf_stats['stat_list'].append(_cf_stats)

	# Making CD profile plots
	print("Making CF profiles...")
	ld.plot_cf_profile(cf_stats, method=cfprof_method, x_origpos=cfprof_x_origpos, var_scale=cfprof_var_scale, xlog=False, ylog=cfprof_ylog, save_formats=save_formats)
	ld.plot_cf_profile(cf_stats, method=cfprof_method, x_origpos=cfprof_x_origpos, var_scale=cfprof_var_scale, xlog=True, ylog=cfprof_ylog, save_formats=save_formats)

	# Release memory
	if wipe_mem:
		del rss


if __name__ == '__main__':	
	sim_directory = utils.get_simus_directory()

	args = parser.parse_args()
	if args.result_battery=="ultra_vs_rb2":
		bit_flips_per_lvl = int(args.bf_ratio*args.artificial_seq_len)
		rs_names = {
			"d{depth_:d}UltraMixed{bf_:d}bits".format(
				depth_ = args.tree_depth,
				bf_ = bit_flips_per_lvl
			): 0,
			"d{depth_:d}UltraUnmixed{bf_:d}bits".format(
				depth_ = args.tree_depth,
				bf_ = bit_flips_per_lvl
			): 0.12,
			"d{depth_:d}RbMixed{bf_:d}bits".format(
				depth_ = args.tree_depth,
				bf_ = bit_flips_per_lvl
			): 0.5,
			"d{depth_:d}RbUnmixed{bf_:d}bits".format(
				depth_ = args.tree_depth,
				bf_ = bit_flips_per_lvl
			): 0.62,
			"d{depth_:d}Unif{bf_:d}bits".format(
				depth_ = args.tree_depth,
				bf_ = bit_flips_per_lvl
			): 0
		}
		accuracy_to_compare = [
			(
				"d{depth_:d}UltraMixed{bf_:d}bits".format(depth_ = args.tree_depth, bf_ = bit_flips_per_lvl),
				"d{depth_:d}RbMixed{bf_:d}bits".format(depth_ = args.tree_depth, bf_ = bit_flips_per_lvl),
				"d{depth_:d}Unif{bf_:d}bits".format(depth_ = args.tree_depth, bf_ = bit_flips_per_lvl)
			),
			(
				"d{depth_:d}UltraUnmixed{bf_:d}bits".format(depth_ = args.tree_depth, bf_ = bit_flips_per_lvl),
				"d{depth_:d}RbUnmixed{bf_:d}bits".format(depth_ = args.tree_depth, bf_ = bit_flips_per_lvl),
				"d{depth_:d}Unif{bf_:d}bits".format(depth_ = args.tree_depth, bf_ = bit_flips_per_lvl)
			),
			(
				"d{depth_:d}UltraMixed{bf_:d}bits".format(depth_ = args.tree_depth, bf_ = bit_flips_per_lvl),
				"d{depth_:d}UltraUnmixed{bf_:d}bits".format(depth_ = args.tree_depth, bf_ = bit_flips_per_lvl),
				"d{depth_:d}Unif{bf_:d}bits".format(depth_ = args.tree_depth, bf_ = self.bit_flips_per_lvl)
			),
			(
				"d{depth_:d}RbMixed{bf_:d}bits".format(depth_ = args.tree_depth, bf_ = bit_flips_per_lvl),
				"d{depth_:d}RbUnmixed{bf_:d}bits".format(depth_ = args.tree_depth, bf_ = bit_flips_per_lvl),
				"d{depth_:d}Unif{bf_:d}bits".format(depth_ = args.tree_depth, bf_ = bit_flips_per_lvl)
			)
		]
		rs_for_lbl_plots = (
			"d{depth_:d}UltraMixed{bf_:d}bits".format(depth_ = args.tree_depth, bf_ = bit_flips_per_lvl),
			"d{depth_:d}RbMixed{bf_:d}bits".format(depth_ = args.tree_depth, bf_ = bit_flips_per_lvl)
		)

	elif args.result_battery=="compare_bit_flipping_mixed":
		rs_names = {}
		accuracy_to_compare = []

		n_bf_ratios = len(bf_ratio_tested)
		for bf_id, bf_ratio in enumerate(bf_ratio_tested):
			bit_flips_per_lvl = int(bf_ratio*args.artificial_seq_len)
			rs["d{depth_:d}UltraMixed{bf_:d}bits".format(depth_ = args.tree_depth, bf_ = bit_flips_per_lvl)] = bf_id/(bf_ratio_tested+0.33)
			rs["d{depth_:d}RbMixed{bf_:d}bits".format(depth_ = args.tree_depth, bf_ = bit_flips_per_lvl)] = (bf_id+0.33)/(bf_ratio_tested+0.33)
			rs["d{depth_:d}Unif{bf_:d}bits".format(depth_ = args.tree_depth, bf_ = bit_flips_per_lvl)] = 0
			
		accuracy_to_compare.append((
			"d{depth_:d}UltraMixed8bits".format(depth_ = args.tree_depth, bf_ = bit_flips_per_lvl),
			"d{depth_:d}UltraMixed20bits".format(depth_ = args.tree_depth, bf_ = bit_flips_per_lvl),
			"d{depth_:d}Unif8bits".format(depth_ = args.tree_depth, bf_ = bit_flips_per_lvl)
		))

		accuracy_to_compare.append((
			"d{depth_:d}UltraMixed14bits".format(depth_ = args.tree_depth, bf_ = bit_flips_per_lvl),
			"d{depth_:d}UltraMixed26bits".format(depth_ = args.tree_depth, bf_ = bit_flips_per_lvl),
			"d{depth_:d}Unif14bits".format(depth_ = args.tree_depth, bf_ = bit_flips_per_lvl)
		))

		accuracy_to_compare.append((
			"d{depth_:d}RbMixed8bits".format(depth_ = args.tree_depth, bf_ = bit_flips_per_lvl),
			"d{depth_:d}RbMixed20bits".format(depth_ = args.tree_depth, bf_ = bit_flips_per_lvl),
			"d{depth_:d}Unif8bits".format(depth_ = args.tree_depth, bf_ = bit_flips_per_lvl)
		))

		accuracy_to_compare.append((
			"d{depth_:d}RbMixed14bits".format(depth_ = args.tree_depth, bf_ = bit_flips_per_lvl),
			"d{depth_:d}RbMixed26bits".format(depth_ = args.tree_depth, bf_ = bit_flips_per_lvl),
			"d{depth_:d}Unif14bits".format(depth_ = args.tree_depth, bf_ = bit_flips_per_lvl)
		))

		rs_for_lbl_plots = ()

	elif args.result_battery=="compare_bit_flipping_unmixed":
		rs_names = {}
		accuracy_to_compare = []

		n_bf_ratios = len(bf_ratio_tested)
		for bf_id, bf_ratio in enumerate(bf_ratio_tested):
			bit_flips_per_lvl = int(bf_ratio*args.artificial_seq_len)
			rs["d{depth_:d}UltraUnmixed{bf_:d}bits".format(depth_ = args.tree_depth, bf_ = bit_flips_per_lvl)] = bf_id/(bf_ratio_tested+0.33)
			rs["d{depth_:d}RbUnmixed{bf_:d}bits".format(depth_ = args.tree_depth, bf_ = bit_flips_per_lvl)] = (bf_id+0.33)/(bf_ratio_tested+0.33)
			rs["d{depth_:d}Unif{bf_:d}bits".format(depth_ = args.tree_depth, bf_ = bit_flips_per_lvl)] = 0
			
		accuracy_to_compare.append((
			"d{depth_:d}UltraUnmixed8bits".format(depth_ = args.tree_depth),
			"d{depth_:d}UltraUnmixed20bits".format(depth_ = args.tree_depth),
			"d{depth_:d}Unif8bits".format(depth_ = args.tree_depth)
		))

		accuracy_to_compare.append((
			"d{depth_:d}UltraUnmixed14bits".format(depth_ = args.tree_depth),
			"d{depth_:d}UltraUnmixed26bits".format(depth_ = args.tree_depth),
			"d{depth_:d}Unif14bits".format(depth_ = args.tree_depth)
		))

		accuracy_to_compare.append((
			"d{depth_:d}RbUnmixed8bits".format(depth_ = args.tree_depth),
			"d{depth_:d}RbUnmixed20bits".format(depth_ = args.tree_depth),
			"d{depth_:d}Unif8bits".format(depth_ = args.tree_depth)
		))

		accuracy_to_compare.append((
			"d{depth_:d}RbUnmixed14bits".format(depth_ = args.tree_depth),
			"d{depth_:d}RbUnmixed26bits".format(depth_ = args.tree_depth),
			"d{depth_:d}Unif14bits".format(depth_ = args.tree_depth)
		))

		rs_for_lbl_plots = ()

	elif args.result_battery=="influence_of_tree_depth_mixed":
		rs_names = {}
		accuracy_to_compare = []

		n_tree_depths = 2
		for depth_id, depth in enumerate((5,7)):
			bit_flips_per_lvl = int(args.bf_ratio*args.artificial_seq_len)
			rs["d{depth_:d}UltraMixed{bf_:d}bits".format(depth_ = depth, bf_ = bit_flips_per_lvl)] = depth_id/(n_tree_depths+0.33)
			rs["d{depth_:d}RbMixed{bf_:d}bits".format(depth_ = depth, bf_ = bit_flips_per_lvl)] = (depth_id+0.33)/(n_tree_depths+0.33)
			rs["d{depth_:d}Unif{bf_:d}bits".format(depth_ = args.tree_depth, bf_ = bit_flips_per_lvl)] = 0
			
		accuracy_to_compare.append((
			"d5UltraMixed{bf_:d}bits".format(bf_ = bit_flips_per_lvl),
			"d7UltraMixed{bf_:d}bits".format(bf_ = bit_flips_per_lvl),
			"d5Unif{bf_:d}bits".format(bf_ = bit_flips_per_lvl)
		))

		accuracy_to_compare.append((
			"d5RbMixed{bf_:d}bits".format(bf_ = bit_flips_per_lvl),
			"d7RbMixed{bf_:d}bits".format(bf_ = bit_flips_per_lvl),
			"d5Unif{bf_:d}bits".format(bf_ = bit_flips_per_lvl)
		))

		rs_for_lbl_plots = ()

	elif args.result_battery=="influence_of_tree_depth_unmixed":
		rs_names = {}
		accuracy_to_compare = []

		n_tree_depths = 2
		for depth_id, depth in enumerate((5,7)):
			bit_flips_per_lvl = int(args.bf_ratio*args.artificial_seq_len)
			rs["d{depth_:d}UltraUnmixed{bf_:d}bits".format(depth_ = depth, bf_ = bit_flips_per_lvl)] = depth_id/(n_tree_depths+0.33)
			rs["d{depth_:d}RbUnmixed{bf_:d}bits".format(depth_ = depth, bf_ = bit_flips_per_lvl)] = (depth_id+0.33)/(n_tree_depths+0.33)
			rs["d{depth_:d}Unif{bf_:d}bits".format(depth_ = args.tree_depth, bf_ = bit_flips_per_lvl)] = 0

		accuracy_to_compare.append((
			"d5UltraUnmixed{bf_:d}bits".format(bf_ = bit_flips_per_lvl),
			"d7UltraUnmixed{bf_:d}bits".format(bf_ = bit_flips_per_lvl),
			"d5Unif{bf_:d}bits".format(bf_ = bit_flips_per_lvl)
		))

		accuracy_to_compare.append((
			"d5RbUnmixed{bf_:d}bits".format(bf_ = bit_flips_per_lvl),
			"d7RbUnmixed{bf_:d}bits".format(bf_ = bit_flips_per_lvl),
			"d5Unif{bf_:d}bits".format(bf_ = bit_flips_per_lvl)
		))

		rs_for_lbl_plots = ()

	print("Making FigureSet object...")
	fs = FigureSet(self, rs_names=rs_names, accuracy_to_compare=accuracy_to_compare, rs_for_lbl_plots=rs_for_lbl_plots, seq_length=300000, n_tests=300, artificial_seq_len=200)
	print("Done")

	print("Making CF-related figures...")
	fig_cfs, ax_cfs = make_CFfigures_artificial(fs, blocks[args.tree_depth], save_formats=['svg', 'pdf'], cfprof_method='mean', cfprof_x_origpos=8.5e4, cfprof_var_scale=0.5, cfprof_ylog=True)
	print("Done")

	if args.make_lbl_history:
		for rs in fs.rs_for_lbl_plots:
			print("Making label history figures...")
			rs.lbl_distrib(max_iter=args.first_iters_focus, cumulative=True, save_formats=['svg', 'pdf'])
			print("Done")