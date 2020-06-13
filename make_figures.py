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
from rs_registry import RS_DIR
utils.format_paper()

## PARAMETERS COMMON TO ALL SIMULATIONS ##
seq_length = 300000
n_tests = 300
artificial_seq_len = 200

artificial_blocks = {
	3: {
		'all': (1, 200, 500, 1000, 2000, 8000, 20000, 40000, 80000),
		'small': (1, 200, 500, 1000),
		'large': (1000, 2000, 8000, 20000, 40000, 80000),
		'acc_plots_shared': (1, 500, 1000, 2000, 20000, 80000),
		'cfhist_plots': (1, 500, 1000, 2000, 20000, 80000)
	},
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

MNIST_blocks = {
	3: {
		'all': (1, 100, 500, 1000, 10000, 100000),
		'small': (1, 100, 500, 1000),
		'large': (1000, 10000, 100000),
		'acc_plots_shared': (1, 100, 500, 1000, 10000, 100000),
		'cfhist_plots': (1, 100, 500, 1000, 10000, 100000)
	}
}

bf_ratio_tested = (0.04, 0.07, 0.1, 0.13)
tree_depths_tested = (5, 7)


## ARGUMENT PARSING
parser = argparse.ArgumentParser('./make_figures.py', description='Produces figures')
parser.add_argument('--dataset', type=str, choices=["artificial", "MNIST"], help="Dataset to produce figures for")
parser.add_argument('--bf_ratio', type=float, help="Bit-flipping ratio per level in the ultrametric tree that generate patterns")
parser.add_argument('--tree_depth', type=int, default=5)

parser.add_argument('--result_battery', type=str, choices=["ultra_vs_rb2", "ultra_vs_rb2_mixed", "ultra_vs_rb2_unmixed", "compare_bit_flipping_mixed", "compare_bit_flipping_unmixed", "influence_of_tree_depth_mixed", "influence_of_tree_depth_unmixed"], help="Battery of results to generate graphs for")
parser.add_argument('--acc_mode', type=str, choices=['unit', 'compare'], default='unif')

parser.add_argument('--make_lbl_history', type=int, default=0, help="Whether to output label history heatmaps and curve-like figures (takes a little time)")
parser.add_argument('--first_iters_focus', type=int, default=50000)


class FigureSet:
	def __init__(self, fs_name, rs_names, accuracy_to_compare, accuracy_plot_style, rs_for_lbl_plots, seq_length=300000, n_tests=300, artificial_seq_len=200):
		self.name = fs_name
		self.seq_length = seq_length
		self.n_tests = n_tests
		self.artificial_seq_len = artificial_seq_len
		self.accuracy_to_compare = accuracy_to_compare
		self.accuracy_plot_style = accuracy_plot_style

		self.load_result_sets(rs_names, rs_for_lbl_plots)

	def load_result_sets(self, rs_names, rs_for_lbl_plots):
		self.rs = {}
		for rs_name, rs_hue in rs_names.items():
			self.rs[rs_name] = RS_DIR[rs_name]
			self.rs[rs_name].load_analytics(hue=rs_hue)
		self.rs_for_lbl_plots = []
		for rs_name in rs_for_lbl_plots:
			self.rs_for_lbl_plots.append(self.rs[rs_name])


def make_CFfigures(fs, blocks, save_formats=['svg', 'pdf'], acc_mode='unit', cfprof_method='mean', cfprof_x_origpos=1.3e5, cfprof_var_scale=1, cfprof_ylog=False, wipe_mem=True):

	## Producing accuracy plots
	print("Making accuracy figures...")
	if fs.accuracy_plot_style == "comp" and acc_mode=='compare':
		for (main_set, altr_set, unif_set) in fs.accuracy_to_compare:
			print("    {:s} vs {:s}".format(main_set, altr_set))
			depth_mainset = fs.rs[main_set].params["Tree Depth"]
			depth_altrset = fs.rs[altr_set].params["Tree Depth"]
			
			ld.make_perfplot_comparison(
				rs=fs.rs[main_set], blocks=blocks[depth_mainset], blocks_altr=blocks[depth_altrset], rs_altr=fs.rs[altr_set], rs_unif=fs.rs[unif_set],
				seq_length=fs.seq_length, n_tests=fs.n_tests,
				blocks_to_plot = 'small', save_formats=save_formats, figset_name=fs_name
			)
			ld.make_perfplot_comparison(
				rs=fs.rs[main_set], blocks=blocks[depth_mainset], blocks_altr=blocks[depth_altrset], rs_altr=fs.rs[altr_set], rs_unif=fs.rs[unif_set],
				seq_length=fs.seq_length, n_tests=fs.n_tests,
				blocks_to_plot = 'large', save_formats=save_formats, figset_name=fs_name
			)

	elif fs.accuracy_plot_style == "comp" and acc_mode=='unit':
		unit_acc_plots = set([(main_set, unif_set) for (main_set, altr_set, unif_set) in fs.accuracy_to_compare] + [(altr_set, unif_set) for (main_set, altr_set, unif_set) in fs.accuracy_to_compare])
		for (main_set, unif_set) in unit_acc_plots:
			print("    {:s} vs unif".format(main_set))
			depth_mainset = fs.rs[main_set].params["Tree Depth"]
		
			ld.make_perfplot_unit(
				rs=fs.rs[main_set], blocks=blocks[depth_mainset], rs_unif=fs.rs[unif_set],
				seq_length=fs.seq_length, n_tests=fs.n_tests,
				blocks_to_plot = 'small', save_formats=save_formats, figset_name=fs_name
			)
			ld.make_perfplot_unit(
				rs=fs.rs[main_set], blocks=blocks[depth_mainset], rs_unif=fs.rs[unif_set],
				seq_length=fs.seq_length, n_tests=fs.n_tests,
				blocks_to_plot = 'large', save_formats=save_formats, figset_name=fs_name
			)

	elif fs.accuracy_plot_style == "matrix":
		for rs_names_list in fs.accuracy_to_compare:
			rs_list = [fs.rs[rs_name] for rs_name in rs_names_list]
			depth = rs_list[0].params["Tree Depth"]
			ld.make_perfplot_matrix(
				rs_list=rs_list, blocks=blocks[depth], rs_unif=None,
				seq_length=fs.seq_length, n_tests=fs.n_tests,
				blocks_to_plot = 'small', save_formats=save_formats, figset_name=fs_name
			)
			ld.make_perfplot_matrix(
				rs_list=rs_list, blocks=blocks[depth], rs_unif=None,
				seq_length=fs.seq_length, n_tests=fs.n_tests,
				blocks_to_plot = 'large', save_formats=save_formats, figset_name=fs_name
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

		avg_cf, avg_cf_std, init_cf, init_cf_std = ld.get_cf_history(rs, blocks[rs.params["Tree Depth"]], save_formats=save_formats, figset_name=fs_name)
		_cf_stats['rs'] = rs
		_cf_stats['rs_name'] = rs.name
		_cf_stats['avg_cf'] = avg_cf
		_cf_stats['avg_cf_std'] = avg_cf_std
		_cf_stats['init_cf'] = init_cf
		_cf_stats['init_cf_std'] = init_cf_std

		fs.cf_stats['stat_list'].append(_cf_stats)

	# Making CD profile plots
	print("Making CF profiles...")
	ld.plot_cf_profile(fs.cf_stats, method=cfprof_method, x_origpos=cfprof_x_origpos, var_scale=cfprof_var_scale, xlog=False, ylog=cfprof_ylog, save_formats=save_formats, figset_name=fs_name)
	ld.plot_cf_profile(fs.cf_stats, method=cfprof_method, x_origpos=cfprof_x_origpos, var_scale=cfprof_var_scale, xlog=True, ylog=cfprof_ylog, save_formats=save_formats, figset_name=fs_name)
	ld.plot_cf_profile(fs.cf_stats, method=cfprof_method, x_origpos=cfprof_x_origpos, var_scale=cfprof_var_scale, xlog=True, ylog=cfprof_ylog, normalize=True, save_formats=save_formats, figset_name=fs_name)

	# Release memory
	if wipe_mem:
		del fs


if __name__ == '__main__':	
	sim_directory = utils.get_simus_directory()

	args = parser.parse_args()
	if args.dataset == 'artificial':
		blocks = artificial_blocks
		if args.result_battery=="ultra_vs_rb2":
			bit_flips_per_lvl = int(args.bf_ratio*artificial_seq_len)
			fs_name = "ultra_vs_rb2_d{depth_:d}_{bf_:d}bits".format(
				depth_ = args.tree_depth,
				bf_ = bit_flips_per_lvl
			)
			rs_names = {
				"artificial_d{depth_:d}UltraMixed{bf_:d}bits".format(
					depth_ = args.tree_depth,
					bf_ = bit_flips_per_lvl
				): 0,
				"artificial_d{depth_:d}UltraUnmixed{bf_:d}bits".format(
					depth_ = args.tree_depth,
					bf_ = bit_flips_per_lvl
				): 0.12,
				"artificial_d{depth_:d}RbMixed{bf_:d}bits".format(
					depth_ = args.tree_depth,
					bf_ = bit_flips_per_lvl
				): 0.5,
				"artificial_d{depth_:d}RbUnmixed{bf_:d}bits".format(
					depth_ = args.tree_depth,
					bf_ = bit_flips_per_lvl
				): 0.62,
				"artificial_d{depth_:d}Unif{bf_:d}bits".format(
					depth_ = args.tree_depth,
					bf_ = bit_flips_per_lvl
				): 0
			}
			accuracy_to_compare = [
				(
					"artificial_d{depth_:d}UltraMixed{bf_:d}bits".format(depth_ = args.tree_depth, bf_ = bit_flips_per_lvl),
					"artificial_d{depth_:d}RbMixed{bf_:d}bits".format(depth_ = args.tree_depth, bf_ = bit_flips_per_lvl),
					"artificial_d{depth_:d}Unif{bf_:d}bits".format(depth_ = args.tree_depth, bf_ = bit_flips_per_lvl)
				),
				(
					"artificial_d{depth_:d}UltraUnmixed{bf_:d}bits".format(depth_ = args.tree_depth, bf_ = bit_flips_per_lvl),
					"artificial_d{depth_:d}RbUnmixed{bf_:d}bits".format(depth_ = args.tree_depth, bf_ = bit_flips_per_lvl),
					"artificial_d{depth_:d}Unif{bf_:d}bits".format(depth_ = args.tree_depth, bf_ = bit_flips_per_lvl)
				),
				(
					"artificial_d{depth_:d}UltraMixed{bf_:d}bits".format(depth_ = args.tree_depth, bf_ = bit_flips_per_lvl),
					"artificial_d{depth_:d}UltraUnmixed{bf_:d}bits".format(depth_ = args.tree_depth, bf_ = bit_flips_per_lvl),
					"artificial_d{depth_:d}Unif{bf_:d}bits".format(depth_ = args.tree_depth, bf_ = bit_flips_per_lvl)
				),
				(
					"artificial_d{depth_:d}RbMixed{bf_:d}bits".format(depth_ = args.tree_depth, bf_ = bit_flips_per_lvl),
					"artificial_d{depth_:d}RbUnmixed{bf_:d}bits".format(depth_ = args.tree_depth, bf_ = bit_flips_per_lvl),
					"artificial_d{depth_:d}Unif{bf_:d}bits".format(depth_ = args.tree_depth, bf_ = bit_flips_per_lvl)
				)
			]
			accuracy_plot_style = "comp"
			rs_for_lbl_plots = (
				"artificial_d{depth_:d}UltraMixed{bf_:d}bits".format(depth_ = args.tree_depth, bf_ = bit_flips_per_lvl),
				"artificial_d{depth_:d}RbMixed{bf_:d}bits".format(depth_ = args.tree_depth, bf_ = bit_flips_per_lvl)
			)

		elif args.result_battery=="ultra_vs_rb2_mixed":
			bit_flips_per_lvl = int(args.bf_ratio*artificial_seq_len)
			fs_name = "ultra_vs_rb2_d{depth_:d}_{bf_:d}bits_mixed".format(
				depth_ = args.tree_depth,
				bf_ = bit_flips_per_lvl
			)
			rs_names = {
				"artificial_d{depth_:d}UltraMixed{bf_:d}bits".format(
					depth_ = args.tree_depth,
					bf_ = bit_flips_per_lvl
				): 0,
				"artificial_d{depth_:d}RbMixed{bf_:d}bits".format(
					depth_ = args.tree_depth,
					bf_ = bit_flips_per_lvl
				): 0.5,
				"artificial_d{depth_:d}Unif{bf_:d}bits".format(
					depth_ = args.tree_depth,
					bf_ = bit_flips_per_lvl
				): 0
			}
			accuracy_to_compare = [
				(
					"artificial_d{depth_:d}UltraMixed{bf_:d}bits".format(depth_ = args.tree_depth, bf_ = bit_flips_per_lvl),
					"artificial_d{depth_:d}RbMixed{bf_:d}bits".format(depth_ = args.tree_depth, bf_ = bit_flips_per_lvl),
					"artificial_d{depth_:d}Unif{bf_:d}bits".format(depth_ = args.tree_depth, bf_ = bit_flips_per_lvl)
				)
			]
			accuracy_plot_style = "comp"
			rs_for_lbl_plots = (
				"artificial_d{depth_:d}UltraMixed{bf_:d}bits".format(depth_ = args.tree_depth, bf_ = bit_flips_per_lvl),
				"artificial_d{depth_:d}RbMixed{bf_:d}bits".format(depth_ = args.tree_depth, bf_ = bit_flips_per_lvl)
			)

		elif args.result_battery=="ultra_vs_rb2_unmixed":
			bit_flips_per_lvl = int(args.bf_ratio*artificial_seq_len)
			fs_name = "ultra_vs_rb2_d{depth_:d}_{bf_:d}bits_unmixed".format(
				depth_ = args.tree_depth,
				bf_ = bit_flips_per_lvl
			)
			rs_names = {
				"artificial_d{depth_:d}UltraUnmixed{bf_:d}bits".format(
					depth_ = args.tree_depth,
					bf_ = bit_flips_per_lvl
				): 0.12,
				"artificial_d{depth_:d}RbUnmixed{bf_:d}bits".format(
					depth_ = args.tree_depth,
					bf_ = bit_flips_per_lvl
				): 0.62,
				"artificial_d{depth_:d}Unif{bf_:d}bits".format(
					depth_ = args.tree_depth,
					bf_ = bit_flips_per_lvl
				): 0
			}
			accuracy_to_compare = [
				(
					"artificial_d{depth_:d}UltraUnmixed{bf_:d}bits".format(depth_ = args.tree_depth, bf_ = bit_flips_per_lvl),
					"artificial_d{depth_:d}RbUnmixed{bf_:d}bits".format(depth_ = args.tree_depth, bf_ = bit_flips_per_lvl),
					"artificial_d{depth_:d}Unif{bf_:d}bits".format(depth_ = args.tree_depth, bf_ = bit_flips_per_lvl)
				)
			]
			rs_for_lbl_plots = (
				"artificial_d{depth_:d}UltraUnmixed{bf_:d}bits".format(depth_ = args.tree_depth, bf_ = bit_flips_per_lvl),
				"artificial_d{depth_:d}RbUnmixed{bf_:d}bits".format(depth_ = args.tree_depth, bf_ = bit_flips_per_lvl)
			)

		elif args.result_battery=="compare_bit_flipping_mixed":
			rs_names = {}
			accuracy_to_compare = []
			accuracy_plot_style = "matrix"
			fs_name = "compare_bit_flipping_mixed_d{depth_:d}".format(
				depth_ = args.tree_depth
			)

			n_bf_ratios = len(bf_ratio_tested)
			for bf_id, bf_ratio in enumerate(bf_ratio_tested):
				bit_flips_per_lvl = int(bf_ratio*artificial_seq_len)
				rs_names["artificial_d{depth_:d}UltraMixed{bf_:d}bits".format(depth_ = args.tree_depth, bf_ = bit_flips_per_lvl)] = bf_id/(n_bf_ratios+0.33)
				rs_names["artificial_d{depth_:d}RbMixed{bf_:d}bits".format(depth_ = args.tree_depth, bf_ = bit_flips_per_lvl)] = (bf_id+0.33)/(n_bf_ratios+0.33)
				rs_names["artificial_d{depth_:d}Unif{bf_:d}bits".format(depth_ = args.tree_depth, bf_ = bit_flips_per_lvl)] = 0
				
			accuracy_to_compare.append((
				"artificial_d{depth_:d}UltraMixed{bf_:d}bits".format(depth_ = args.tree_depth, bf_ = int(bf_ratio*artificial_seq_len)) for bf_ratio in bf_ratio_tested
			))

			accuracy_to_compare.append((
				"artificial_d{depth_:d}RbMixed{bf_:d}bits".format(depth_ = args.tree_depth, bf_ = int(bf_ratio*artificial_seq_len)) for bf_ratio in bf_ratio_tested
			))

			rs_for_lbl_plots = ()

		elif args.result_battery=="compare_bit_flipping_unmixed":
			rs_names = {}
			accuracy_to_compare = []
			accuracy_plot_style = "matrix"
			fs_name = "compare_bit_flipping_unmixed_d{depth_:d}".format(
				depth_ = args.tree_depth
			)

			n_bf_ratios = len(bf_ratio_tested)
			for bf_id, bf_ratio in enumerate(bf_ratio_tested):
				bit_flips_per_lvl = int(bf_ratio*artificial_seq_len)
				rs_names["artificial_d{depth_:d}UltraUnmixed{bf_:d}bits".format(depth_ = args.tree_depth, bf_ = bit_flips_per_lvl)] = bf_id/(n_bf_ratios+0.33)
				rs_names["artificial_d{depth_:d}RbUnmixed{bf_:d}bits".format(depth_ = args.tree_depth, bf_ = bit_flips_per_lvl)] = (bf_id+0.33)/(n_bf_ratios+0.33)
				rs_names["artificial_d{depth_:d}Unif{bf_:d}bits".format(depth_ = args.tree_depth, bf_ = bit_flips_per_lvl)] = 0
				
			accuracy_to_compare.append((
				"artificial_d{depth_:d}UltraUnmixed{bf_:d}bits".format(depth_ = args.tree_depth, bf_ = bf) for bf in bf_ratio_tested
			))

			accuracy_to_compare.append((
				"artificial_d{depth_:d}RbUnmixed{bf_:d}bits".format(depth_ = args.tree_depth, bf_ = bf) for bf in bf_ratio_tested
			))

			rs_for_lbl_plots = ()

		elif args.result_battery=="influence_of_tree_depth_mixed":
			rs_names = {}
			accuracy_to_compare = []
			accuracy_plot_style = "comp"
			bit_flips_per_lvl = int(args.bf_ratio*artificial_seq_len)
			fs_name = "influence_of_tree_depth_mixed_{bf_:d}bits".format(
				bf_ = bit_flips_per_lvl
			)

			n_tree_depths = 2
			for depth_id, depth in enumerate((5,7)):
				rs_names["artificial_d{depth_:d}UltraMixed{bf_:d}bits".format(depth_ = depth, bf_ = bit_flips_per_lvl)] = depth_id/(n_tree_depths+0.33)
				rs_names["artificial_d{depth_:d}RbMixed{bf_:d}bits".format(depth_ = depth, bf_ = bit_flips_per_lvl)] = (depth_id+0.33)/(n_tree_depths+0.33)
				rs_names["artificial_d{depth_:d}Unif{bf_:d}bits".format(depth_ = args.tree_depth, bf_ = bit_flips_per_lvl)] = 0
				
			accuracy_to_compare.append((
				"artificial_d5UltraMixed{bf_:d}bits".format(bf_ = bit_flips_per_lvl),
				"artificial_d7UltraMixed{bf_:d}bits".format(bf_ = bit_flips_per_lvl),
				"artificial_d5Unif{bf_:d}bits".format(bf_ = bit_flips_per_lvl)
			))

			accuracy_to_compare.append((
				"artificial_d5RbMixed{bf_:d}bits".format(bf_ = bit_flips_per_lvl),
				"artificial_d7RbMixed{bf_:d}bits".format(bf_ = bit_flips_per_lvl),
				"artificial_d5Unif{bf_:d}bits".format(bf_ = bit_flips_per_lvl)
			))

			rs_for_lbl_plots = ()

		elif args.result_battery=="influence_of_tree_depth_unmixed":
			rs_names = {}
			accuracy_to_compare = []
			accuracy_plot_style = "comp"
			bit_flips_per_lvl = int(args.bf_ratio*artificial_seq_len)
			fs_name = "influence_of_tree_depth_unmixed_{bf_:d}bits".format(
				bf_ = bit_flips_per_lvl
			)

			n_tree_depths = 2
			for depth_id, depth in enumerate((5,7)):
				rs_names["artificial_d{depth_:d}UltraUnmixed{bf_:d}bits".format(depth_ = depth, bf_ = bit_flips_per_lvl)] = depth_id/(n_tree_depths+0.33)
				rs_names["artificial_d{depth_:d}RbUnmixed{bf_:d}bits".format(depth_ = depth, bf_ = bit_flips_per_lvl)] = (depth_id+0.33)/(n_tree_depths+0.33)
				rs_names["artificial_d{depth_:d}Unif{bf_:d}bits".format(depth_ = args.tree_depth, bf_ = bit_flips_per_lvl)] = 0

			accuracy_to_compare.append((
				"artificial_d5UltraUnmixed{bf_:d}bits".format(bf_ = bit_flips_per_lvl),
				"artificial_d7UltraUnmixed{bf_:d}bits".format(bf_ = bit_flips_per_lvl),
				"artificial_d5Unif{bf_:d}bits".format(bf_ = bit_flips_per_lvl)
			))

			accuracy_to_compare.append((
				"artificial_d5RbUnmixed{bf_:d}bits".format(bf_ = bit_flips_per_lvl),
				"artificial_d7RbUnmixed{bf_:d}bits".format(bf_ = bit_flips_per_lvl),
				"artificial_d5Unif{bf_:d}bits".format(bf_ = bit_flips_per_lvl)
			))

			rs_for_lbl_plots = ()

	elif args.dataset == 'MNIST':
		blocks = MNIST_blocks
		if args.result_battery=="ultra_vs_rb2":
			fs_name = "mlp_sgd"
			rs_names = {
				"MNIST_UltraMixed": 0,
				"MNIST_RbMixed": 0.5,
				"MNIST_Unif": 0
			}
			accuracy_to_compare = [
				(
					"MNIST_UltraMixed",
					"MNIST_RbMixed",
					"MNIST_Unif"
				)
			]
			accuracy_plot_style = "comp"
			rs_for_lbl_plots = (
				"MNIST_UltraMixed",
				"MNIST_RbMixed"
			)

	fs_name = "{dataset:s}/{fs_base:s}".format(
		dataset=args.dataset,
		fs_base=fs_name
	)

	print("Making FigureSet object...")
	fs = FigureSet(fs_name=fs_name, rs_names=rs_names, accuracy_to_compare=accuracy_to_compare, accuracy_plot_style=accuracy_plot_style, rs_for_lbl_plots=rs_for_lbl_plots, seq_length=300000, n_tests=300, artificial_seq_len=200)
	print("Done")

	if args.make_lbl_history < 2:
		print("Making CF-related figures...")
		make_CFfigures(fs, blocks, save_formats=['svg', 'pdf'], acc_mode=args.acc_mode, cfprof_method='mean', cfprof_x_origpos=8.5e4, cfprof_var_scale=0.5, cfprof_ylog=False)
		print("Done")

	if args.make_lbl_history:
		for rs in fs.rs_for_lbl_plots:
			print("Making label history figures...")
			rs.lbl_distrib(max_iter=args.first_iters_focus, cumulative=True, save_formats=['svg', 'pdf'], multi_simus=False)
			rs.lbl_distrib(max_iter=args.first_iters_focus, cumulative=False, save_formats=['svg', 'pdf'], multi_simus=False)
			print("Done")