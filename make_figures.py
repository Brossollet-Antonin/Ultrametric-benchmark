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
from rs_registry import RS_DIR, get_directory_view
utils.format_paper()

## PARAMETERS COMMON TO ALL SIMULATIONS ##
artificial_blocks = {
	3: {
		'all': (1, 200, 500, 1000, 2000, 8000, 20000, 40000, 80000),
		'small': (1, 200, 500, 1000),
		'large': (1000, 2000, 8000, 20000, 40000, 80000),
		'synth': (1, 200, 1000, 4000, 8000, 20000),
		'cfhist_plots': (1, 500, 1000, 2000, 20000, 80000)
	},
	4: { # NONLINEAR CASE
		'all': (1, 25, 50, 75, 100, 150, 300, 600, 1200, 2400, 4800, 9600, 19200, 38400, 76800, 153600),
		'small': (1, 25, 75, 150),
		'large': (150, 300, 600, 1200, 2400, 4800, 9600, 19200, 38400, 76800, 153600),
		'synth': (1, 25, 150, 1200, 9600, 38400),
		'cfhist_plots': (1, 25, 150, 1200, 9600, 38400)
	},
	# 4: { # LINEAR CASE
	# 	'all': (1, 25, 75, 150, 300, 600, 1200, 2400, 4800, 10800, 19200, 39600, 79200),
	# 	'small': (1, 25, 75, 150),
	# 	'large': (150, 300, 600, 1200, 2400, 4800, 10800, 19200, 39600, 79200),
	# 	'synth': (1, 25, 150, 1200, 10800, 39600),
	# 	'cfhist_plots': (1, 25, 150, 1200, 10800, 39600)
	# },
	5: { # NONLINEAR CASE
	 	'all': (1, 200, 400, 500, 600, 800, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 256000),
	 	'small': (1, 200, 500, 1000),
	 	'large': (1000, 2000, 8000, 16000, 32000, 64000, 128000, 256000),
	 	'synth': (1, 200, 1000, 4000, 16000, 64000),
	 	'cfhist_plots': (1, 200, 1000, 4000, 16000, 64000)
	},
	# 5: { # LINEAR CASE
	#	'all': (1, 100, 200, 500, 1000, 2000, 4000, 6000, 8000, 10000, 20000, 40000, 80000),
	#	'small': (1, 100, 200, 500, 1000),
	#	'large': (1000, 2000, 4000, 6000, 8000, 10000, 20000, 40000, 80000),
	#	'synth': (1, 200, 1000, 4000, 20000, 80000),
	#	'cfhist_plots': (1, 200, 1000, 4000, 20000, 80000)
	# },
	6: { # NONLINEAR CASE
	 	'all': (1, 500, 1000, 2500, 5000, 10000, 20000, 40000, 80000, 160000, 320000, 640000, 1280000),
	 	'small': (1, 500, 1000, 2500, 5000),
	 	'large': (5000, 10000, 20000, 40000, 80000, 160000, 320000, 640000, 1280000),
	 	'synth': (1, 1000, 5000, 40000, 160000, 640000),
	 	'cfhist_plots': (1, 1000, 5000, 40000, 160000, 640000)
	},
	# 6: { # LINEAR CASE
	#	'all': (1, 125, 250, 500, 1000, 2500, 5000, 10000, 20000, 40000, 80000),
	#	'small': (1, 500, 1000, 2500),
	#	'large': (2500, 5000, 10000, 20000, 40000, 80000),
	#	'synth': (1, 250, 2500, 5000, 20000, 80000),
	#	'cfhist_plots': (1, 250, 2500, 5000, 20000, 80000)
	# },
	7: {
		'all': (1, 82, 164, 328, 1312, 5248, 20992, 41984, 83968),
		'small': (1, 82, 164, 328),
		'large': (328, 1312, 5248, 20992, 41984, 83968),
		'synth': (1, 164, 328, 1312, 20992, 83968),
		'cfhist_plots': (1, 328, 1312, 20992, 83968)
	},
	10: {
		'all': (1,),
		'small': (1,),
		'large': (1,),
		'synth': (1,),
		'cfhist_plots': (1,)
	}
}

MNIST_blocks = {
	3: {
		'all': (1, 180, 360, 720, 1440, 2880, 5760, 11520, 23040, 46080, 92160, 184320, 368640, 737280, 1474560),
		'small': (1, 180, 360),
		'large': (720, 1440, 2880, 5760, 11520, 23040, 46080, 92160, 184320, 368640, 737280, 1474560),
		'synth': (1, 360, 2880, 23040, 184320, 1474560),
		'cfhist_plots': (1, 360, 2880, 23040, 184320, 1474560)
	}
}

bf_ratio_tested = (0.04, 0.07, 0.1, 0.13)
tree_depths_tested = range(3, 8)


## ARGUMENT PARSING
parser = argparse.ArgumentParser('./make_figures.py', description='Produces figures')
parser.add_argument('--see_rsdir', action='store_true', help="See available result sets")

parser.add_argument('--dataset', type=str, choices=["artificial", "MNIST"], help="Dataset to produce figures for")
parser.add_argument('--bf_ratio', type=float, help="Bit-flipping ratio per level in the ultrametric tree that generate patterns")
parser.add_argument('--tree_depth', type=int, default=5)
parser.add_argument('--optimizer', type=str, default="sgd")

parser.add_argument('--artificial_seq_len', type=int, default=200, help="In the case of the artificial dataset, length of each patterns used for generating exemplars")

parser.add_argument('--result_battery', type=str, choices=["ultra_vs_rb2", "ultra_vs_rb2_mixed", "ultra_vs_rb2_unmixed", "compare_bit_flipping_mixed", "compare_bit_flipping_unmixed", "influence_of_tree_depth_mixed", "influence_of_tree_depth_unmixed", "influence_of_tree_depth_test", "optim_tryout"], help="Battery of results to generate graphs for")
parser.add_argument('--acc_mode', type=str, choices=['unit', 'compare'], default='unit')
parser.add_argument('--draw_timescales', action='store_true', help='enable to draw ultrametric tree timescales on top of classificaiton accuracy plots')
parser.add_argument('--draw_explorations', action='store_true', help='enable to draw ultrametric tree timescales on top of classificaiton accuracy plots')

parser.add_argument('--make_lbl_history', type=int, default=0, help="Whether to output label history heatmaps and curve-like figures (takes a little time)")
parser.add_argument('--first_iters_focus', type=int, default=50000)

parser.add_argument('--cfprof_method', type=str, default="raw", choices=["raw", "aligned"], help="Method used for measuring catastrophic forgetting")
parser.add_argument('--cfprof_x_origpos', type=int, default=2e5)
parser.add_argument('--acc_max_iter', type=int, default=None)
parser.add_argument('--cfprof_ymax', type=float, default=None)
parser.add_argument('--cf_confidence', type=float, default=0.05)
parser.add_argument('--rb2_norm', action='store_true', default=False)
parser.add_argument('--fit_um_profile', default='none', choices=["none", "sigmoid", "sigmoid_primitive"])
parser.add_argument('--fit_rb2_profile', default='none', choices=["none", "sigmoid", "sigmoid_primitive", "cut_sigmoid", "cut_nonlinear_sigmoid"])


class FigureSet:
	def __init__(self, fs_name, rs_names, accuracy_to_compare, accuracy_plot_style, rs_for_lbl_plots, artificial_seq_len=200, cf_correctionfactor=None, cfprof_method="mean", draw_timescales=False, draw_explorations=False):
		"""
		Correction factor: variable that allows to compensate catastrophic forgetting when we don't have the entire learning curve, for instance when learning is too slow so that our samp le stops before 100% accuracy
		"""
		self.name = fs_name
		self.artificial_seq_len = artificial_seq_len
		self.accuracy_to_compare = accuracy_to_compare
		self.accuracy_plot_style = accuracy_plot_style
		self.cf_correctionfactor = cf_correctionfactor
		self.draw_timescales = draw_timescales
		self.draw_explorations = draw_explorations

		self.load_result_sets(rs_names, rs_for_lbl_plots)

	def load_result_sets(self, rs_names, rs_for_lbl_plots):
		self.rs = {}
		for rs_name, rs_hue in rs_names.items():
			self.rs[rs_name] = RS_DIR[rs_name]
			self.rs[rs_name].load_analytics(hue=rs_hue)
		self.rs_for_lbl_plots = []
		for rs_name in rs_for_lbl_plots:
			self.rs_for_lbl_plots.append(self.rs[rs_name])


def make_CFfigures(fs, blocks, save_formats=['svg', 'pdf'], blocksets_to_plot=['synth'], acc_mode='unit', acc_max_iter=None, cfprof_x_origpos=1.3e5, cf_conf=0.05, cfprof_ymax=None, rb2_norm=True, fit_um_profile='none', fit_rb2_profile='none', wipe_mem=True):

	if acc_max_iter is None:
		plot_window = None
	else:
		plot_window	= (0, acc_max_iter)

	## Producing accuracy plots
	print("Making accuracy figures...")
	if fs.accuracy_plot_style == "comp" and acc_mode=='compare':
		for (main_set, altr_set, unif_set) in fs.accuracy_to_compare:
			print("    {:s} vs {:s}".format(main_set, altr_set))
			depth_mainset = fs.rs[main_set].params["Tree Depth"]
			depth_altrset = fs.rs[altr_set].params["Tree Depth"]
			
			for blockset_to_plot in blocksets_to_plot:
				ld.make_perfplot_comparison(
					rs=fs.rs[main_set], blocks=blocks[depth_mainset], blocks_altr=blocks[depth_altrset], rs_altr=fs.rs[altr_set], rs_unif=fs.rs[unif_set],
					blocks_to_plot = blockset_to_plot, plot_window=plot_window, save_formats=save_formats, figset_name=fs_name, draw_timescales=fs.draw_timescales, draw_explorations=fs.draw_explorations
				)

	elif fs.accuracy_plot_style == "comp" and acc_mode=='unit':
		unit_acc_plots = set([(main_set, unif_set) for (main_set, altr_set, unif_set) in fs.accuracy_to_compare] + [(altr_set, unif_set) for (main_set, altr_set, unif_set) in fs.accuracy_to_compare])
		for (main_set, unif_set) in unit_acc_plots:
			print("    {:s} vs unif".format(main_set))
			depth_mainset = fs.rs[main_set].params["Tree Depth"]
		
			for blockset_to_plot in blocksets_to_plot:
				ld.make_perfplot_unit(
					rs=fs.rs[main_set], blocks=blocks[depth_mainset], rs_unif=fs.rs[unif_set],
					blocks_to_plot = blockset_to_plot, plot_window=plot_window, save_formats=save_formats, figset_name=fs_name, draw_timescales=fs.draw_timescales, draw_explorations=fs.draw_explorations
				)

	elif fs.accuracy_plot_style == "matrix":
		for rs_names_list in fs.accuracy_to_compare:
			rs_list = [fs.rs[rs_name] for rs_name in rs_names_list]
			depth = rs_list[0].params["Tree Depth"]
			for blockset_to_plot in blocksets_to_plot:
				ld.make_perfplot_matrix(
					rs_list=rs_list, blocks=blocks[depth], rs_unif=None,
					blocks_to_plot = blockset_to_plot, plot_window=plot_window, save_formats=save_formats, figset_name=fs_name, draw_timescales=fs.draw_timescales, draw_explorations=fs.draw_explorations
				)

	# Making CF history plots
	fs.cf_stats = {}

	print("Computing CF histories")
	for rs_id, (rs_name, rs) in enumerate(fs.rs.items()):
		print("    {:s}".format(rs_name))
		if ("Unif" in rs_name):
			continue
		_cf_stats = {}

		if fs.cf_correctionfactor is not None:
			cf_correctionfactor = fs.cf_correctionfactor[rs_id]
		else:
			cf_correctionfactor = None

		tot_cf, tot_cf_ci, avg_cf, avg_cf_ci, tot_aligned_cf, tot_aligned_cf_ci, avg_aligned_cf, avg_aligned_cf_ci = ld.get_cf_history(rs, blocks[rs.params["Tree Depth"]], confidence=cf_conf, save_formats=save_formats, figset_name=fs_name, cf_correctionfactor=cf_correctionfactor)
		_cf_stats['rs'] = rs
		_cf_stats['data'] = {}
		
		_cf_stats['data']['tot_ald_cf'] = {k: max(0,v) for k,v in tot_aligned_cf.items()}
		_cf_stats['data']['tot_ald_cf_ci'] = tot_aligned_cf_ci
		_cf_stats['data']['avg_ald_cf'] = {k: max(0,v) for k,v in avg_aligned_cf.items()}
		_cf_stats['data']['avg_ald_cf_ci'] = avg_aligned_cf_ci

		_cf_stats['data']['tot_raw_cf'] = {k: max(0,v) for k,v in tot_cf.items()}
		_cf_stats['data']['tot_raw_cf_ci'] = tot_cf_ci
		_cf_stats['data']['avg_raw_cf'] = {k: max(0,v) for k,v in avg_cf.items()}
		_cf_stats['data']['avg_raw_cf_ci'] = avg_cf_ci

		# Fitting curves to PLTS profiles
		if "Ultra" in rs_name and fit_um_profile!="none":
			ld.fit_profile(_cf_stats, version=fit_um_profile)

		if "Rb" in rs_name and fit_rb2_profile!="none":
			ld.fit_profile(_cf_stats, version=fit_rb2_profile)

		fs.cf_stats[rs.name] = _cf_stats

	if "MNIST" in fs.name:
		ld.report_steepness(fs.cf_stats, depths=[3], metric="avg", ylog_steepness=False, ylog_maxval=False, normalize=False, confidence=cf_conf, save_formats=save_formats, figset_name=fs_name)
		ld.report_steepness(fs.cf_stats, depths=[3], metric="tot", ylog_steepness=False, ylog_maxval=True, normalize=False, confidence=cf_conf, save_formats=save_formats, figset_name=fs_name)

	if "tree_depth" in fs.name:
		bf = int(args.bf_ratio*args.artificial_seq_len)
		ld.report_steepness(fs.cf_stats, depths=[4,5,6], metric="avg", ylog_steepness=False, ylog_maxval=False, normalize=False, bf=bf, confidence=cf_conf, save_formats=save_formats, figset_name=fs_name)
		#ld.report_steepness(fs.cf_stats, depths=[4,5,6], metric="avg", normalize=False, bf=bf, confidence=cf_conf, save_formats=save_formats, figset_name=fs_name)
		ld.report_steepness(fs.cf_stats, depths=[4,5,6], metric="tot", ylog_steepness=False, ylog_maxval=True, normalize=False, bf=bf, confidence=cf_conf, save_formats=save_formats, figset_name=fs_name)
		#ld.report_steepness(fs.cf_stats, depths=[4,5,6], metric="tot", normalize=False, bf=bf, confidence=cf_conf, save_formats=save_formats, figset_name=fs_name)

		if rb2_norm:
			#### NORMALIZES THE CF PROFILES OF THE ULTRAMETRIC SEQUENCES TO THOSE THE CORRESPONDING RANDOM BLOCKS SCENARIOS ####
			for depth in [4,5,6]:
				rs_um_name = "artificial_d{depth_:d}UltraMixed{bf_:d}bits".format(depth_ = depth, bf_ = bf)
				rs_rb_name = "artificial_d{depth_:d}RbMixed{bf_:d}bits".format(depth_ = depth, bf_ = bf)
				
				fs.cf_stats[rs_um_name]['data']['tot_raw_cf'] = {k: v/fs.cf_stats[rs_rb_name]['data']['tot_raw_cf'][k] for k,v in fs.cf_stats[rs_um_name]['data']['tot_raw_cf'].items() if fs.cf_stats[rs_rb_name]['data']['tot_raw_cf'][k]>0}
				fs.cf_stats[rs_um_name]['data']['avg_raw_cf'] = {k: v/fs.cf_stats[rs_rb_name]['data']['avg_raw_cf'][k] for k,v in fs.cf_stats[rs_um_name]['data']['avg_raw_cf'].items() if fs.cf_stats[rs_rb_name]['data']['avg_raw_cf'][k]>0}
				fs.cf_stats[rs_um_name]['data']['tot_raw_cf_ci'] = {k: v/fs.cf_stats[rs_rb_name]['data']['tot_raw_cf'][k] for k,v in fs.cf_stats[rs_um_name]['data']['tot_raw_cf_ci'].items() if fs.cf_stats[rs_rb_name]['data']['tot_raw_cf'][k]>0}
				fs.cf_stats[rs_um_name]['data']['avg_raw_cf_ci'] = {k: v/fs.cf_stats[rs_rb_name]['data']['avg_raw_cf'][k] for k,v in fs.cf_stats[rs_um_name]['data']['avg_raw_cf_ci'].items() if fs.cf_stats[rs_rb_name]['data']['avg_raw_cf'][k]>0}

				fs.cf_stats[rs_um_name]['data']['tot_ald_cf'] = {k: v/fs.cf_stats[rs_rb_name]['data']['tot_ald_cf'][k] for k,v in fs.cf_stats[rs_um_name]['data']['tot_ald_cf'].items() if fs.cf_stats[rs_rb_name]['data']['tot_ald_cf'][k]>0}
				fs.cf_stats[rs_um_name]['data']['avg_ald_cf'] = {k: v/fs.cf_stats[rs_rb_name]['data']['avg_ald_cf'][k] for k,v in fs.cf_stats[rs_um_name]['data']['avg_ald_cf'].items() if fs.cf_stats[rs_rb_name]['data']['avg_ald_cf'][k]>0}
				fs.cf_stats[rs_um_name]['data']['tot_ald_cf_ci'] = {k: v/fs.cf_stats[rs_rb_name]['data']['tot_ald_cf'][k] for k,v in fs.cf_stats[rs_um_name]['data']['tot_ald_cf_ci'].items() if fs.cf_stats[rs_rb_name]['data']['tot_ald_cf'][k]>0}
				fs.cf_stats[rs_um_name]['data']['avg_ald_cf_ci'] = {k: v/fs.cf_stats[rs_rb_name]['data']['avg_ald_cf'][k] for k,v in fs.cf_stats[rs_um_name]['data']['avg_ald_cf_ci'].items() if fs.cf_stats[rs_rb_name]['data']['avg_ald_cf'][k]>0}
				
				del fs.cf_stats[rs_rb_name]
				if 'model_fit' in fs.cf_stats[rs_um_name].keys():
					fs.cf_stats[rs_um_name]['model_fit'] = {}

		del fs.cf_stats["artificial_d5UltraMixed20bits"]["data"]["tot_raw_cf"][200]
		del fs.cf_stats["artificial_d5UltraMixed20bits"]["data"]["tot_raw_cf"][500]
		del fs.cf_stats["artificial_d5UltraMixed20bits"]["data"]["avg_raw_cf"][200]
		del fs.cf_stats["artificial_d5UltraMixed20bits"]["data"]["avg_raw_cf"][500]
		del fs.cf_stats["artificial_d5UltraMixed20bits"]["data"]["tot_ald_cf"][200]
		del fs.cf_stats["artificial_d5UltraMixed20bits"]["data"]["tot_ald_cf"][500]
		del fs.cf_stats["artificial_d5UltraMixed20bits"]["data"]["avg_ald_cf"][200]
		del fs.cf_stats["artificial_d5UltraMixed20bits"]["data"]["avg_ald_cf"][500]

		del fs.cf_stats["artificial_d6UltraMixed20bits"]["data"]["tot_raw_cf"][1000]
		del fs.cf_stats["artificial_d6UltraMixed20bits"]["data"]["tot_raw_cf"][2500]
		del fs.cf_stats["artificial_d6UltraMixed20bits"]["data"]["avg_raw_cf"][1000]
		del fs.cf_stats["artificial_d6UltraMixed20bits"]["data"]["avg_raw_cf"][2500]
		del fs.cf_stats["artificial_d6UltraMixed20bits"]["data"]["tot_ald_cf"][1000]
		del fs.cf_stats["artificial_d6UltraMixed20bits"]["data"]["tot_ald_cf"][2500]
		del fs.cf_stats["artificial_d6UltraMixed20bits"]["data"]["avg_ald_cf"][1000]
		del fs.cf_stats["artificial_d6UltraMixed20bits"]["data"]["avg_ald_cf"][2500]

		####################################################################################################################

	# Making CD profile plots
	print("Making CF profiles...")

	for alignment_method in ["raw", "aligned"]:
		for metric in ["mean", "total"]:
			ld.plot_cf_profile(fs.cf_stats, alignment_method=alignment_method, metric=metric, normalize=False, x_origpos=cfprof_x_origpos, xlog=True, ylog=False, cfprof_ymax=cfprof_ymax, plot_timescales=True, save_formats=save_formats, figset_name=fs_name)

	# Release memory
	if wipe_mem:
		del fs


if __name__ == '__main__':	
	sim_directory = utils.get_simus_directory()

	args = parser.parse_args()
	if args.see_rsdir is True:
		get_directory_view()
	if args.dataset == 'artificial':
		blocks = artificial_blocks
		if args.result_battery=="ultra_vs_rb2":
			bit_flips_per_lvl = int(args.bf_ratio*args.artificial_seq_len)
			fs_name = "ultra_vs_rb2_d{depth_:d}_{bf_:d}bits".format(
				depth_ = args.tree_depth,
				bf_ = bit_flips_per_lvl
			)
			rs_names = {
				"artificial_d{depth_:d}UltraMixed{bf_:d}bits".format(
					depth_ = args.tree_depth,
					bf_ = bit_flips_per_lvl
				): 0.6,
				"artificial_d{depth_:d}UltraUnmixed{bf_:d}bits".format(
					depth_ = args.tree_depth,
					bf_ = bit_flips_per_lvl
				): 0.72,
				"artificial_d{depth_:d}RbMixed{bf_:d}bits".format(
					depth_ = args.tree_depth,
					bf_ = bit_flips_per_lvl
				): 0.3,
				"artificial_d{depth_:d}RbUnmixed{bf_:d}bits".format(
					depth_ = args.tree_depth,
					bf_ = bit_flips_per_lvl
				): 0.42,
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
			cf_correctionfactor = None

		elif args.result_battery=="ultra_vs_rb2_mixed":
			bit_flips_per_lvl = int(args.bf_ratio*args.artificial_seq_len)
			fs_name = "ultra_vs_rb2_d{depth_:d}_{bf_:d}bits_mixed".format(
				depth_ = args.tree_depth,
				bf_ = bit_flips_per_lvl
			)
			rs_names = {
				"artificial_d{depth_:d}UltraMixed{bf_:d}bits".format(
					depth_ = args.tree_depth,
					bf_ = bit_flips_per_lvl
				): 0.6,
				"artificial_d{depth_:d}RbMixed{bf_:d}bits".format(
					depth_ = args.tree_depth,
					bf_ = bit_flips_per_lvl
				): 0.3,
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
			cf_correctionfactor = None

		elif args.result_battery=="ultra_vs_rb2_unmixed":
			bit_flips_per_lvl = int(args.bf_ratio*args.artificial_seq_len)
			fs_name = "ultra_vs_rb2_d{depth_:d}_{bf_:d}bits_unmixed".format(
				depth_ = args.tree_depth,
				bf_ = bit_flips_per_lvl
			)
			rs_names = {
				"artificial_d{depth_:d}UltraUnmixed{bf_:d}bits".format(
					depth_ = args.tree_depth,
					bf_ = bit_flips_per_lvl
				): 0.72,
				"artificial_d{depth_:d}RbUnmixed{bf_:d}bits".format(
					depth_ = args.tree_depth,
					bf_ = bit_flips_per_lvl
				): 0.42,
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
			accuracy_plot_style = "comp"
			rs_for_lbl_plots = (
				"artificial_d{depth_:d}UltraUnmixed{bf_:d}bits".format(depth_ = args.tree_depth, bf_ = bit_flips_per_lvl),
				"artificial_d{depth_:d}RbUnmixed{bf_:d}bits".format(depth_ = args.tree_depth, bf_ = bit_flips_per_lvl)
			)
			cf_correctionfactor = None

		elif args.result_battery=="compare_bit_flipping_mixed":
			rs_names = {}
			accuracy_to_compare = []
			accuracy_plot_style = "matrix"
			fs_name = "compare_bit_flipping_mixed_d{depth_:d}".format(
				depth_ = args.tree_depth
			)

			n_bf_ratios = len(bf_ratio_tested)
			for bf_id, bf_ratio in enumerate(bf_ratio_tested):
				bit_flips_per_lvl = int(bf_ratio*args.artificial_seq_len)
				rs_names["artificial_d{depth_:d}UltraMixed{bf_:d}bits".format(depth_ = args.tree_depth, bf_ = bit_flips_per_lvl)] = bf_id/(n_bf_ratios+0.33)
				rs_names["artificial_d{depth_:d}RbMixed{bf_:d}bits".format(depth_ = args.tree_depth, bf_ = bit_flips_per_lvl)] = (bf_id+0.33)/(n_bf_ratios+0.33)
				rs_names["artificial_d{depth_:d}Unif{bf_:d}bits".format(depth_ = args.tree_depth, bf_ = bit_flips_per_lvl)] = 0
				
			accuracy_to_compare.append((
				"artificial_d{depth_:d}UltraMixed{bf_:d}bits".format(depth_ = args.tree_depth, bf_ = int(bf_ratio*args.artificial_seq_len)) for bf_ratio in bf_ratio_tested
			))

			accuracy_to_compare.append((
				"artificial_d{depth_:d}RbMixed{bf_:d}bits".format(depth_ = args.tree_depth, bf_ = int(bf_ratio*args.artificial_seq_len)) for bf_ratio in bf_ratio_tested
			))

			rs_for_lbl_plots = ()
			cf_correctionfactor = None

		elif args.result_battery=="compare_bit_flipping_unmixed":
			rs_names = {}
			accuracy_to_compare = []
			accuracy_plot_style = "matrix"
			fs_name = "compare_bit_flipping_unmixed_d{depth_:d}".format(
				depth_ = args.tree_depth
			)

			n_bf_ratios = len(bf_ratio_tested)
			for bf_id, bf_ratio in enumerate(bf_ratio_tested):
				bit_flips_per_lvl = int(bf_ratio*args.artificial_seq_len)
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
			cf_correctionfactor = None

		elif args.result_battery=="influence_of_tree_depth_mixed":
			rs_names = {}
			accuracy_to_compare = []
			accuracy_plot_style = "comp"
			bit_flips_per_lvl = int(args.bf_ratio*args.artificial_seq_len)
			fs_name = "influence_of_tree_depth_mixed_{bf_:d}bits".format(
				bf_ = bit_flips_per_lvl
			)

			n_tree_depths = 3
			for depth_id, depth in enumerate((4,5,6)):
				rs_names["artificial_d{depth_:d}UltraMixed{bf_:d}bits".format(depth_ = depth, bf_ = bit_flips_per_lvl)] = depth_id/(n_tree_depths+0.33)
				rs_names["artificial_d{depth_:d}RbMixed{bf_:d}bits".format(depth_ = depth, bf_ = bit_flips_per_lvl)] = (depth_id+0.33)/(n_tree_depths+0.33)
				rs_names["artificial_d{depth_:d}Unif{bf_:d}bits".format(depth_ = depth, bf_ = bit_flips_per_lvl)] = 0
				
				accuracy_to_compare.append((
					"artificial_d{depth_:d}RbMixed{bf_:d}bits".format(depth_ = depth, bf_ = bit_flips_per_lvl),
					"artificial_d{depth_:d}UltraMixed{bf_:d}bits".format(depth_ = depth, bf_ = bit_flips_per_lvl),
					"artificial_d{depth_:d}Unif{bf_:d}bits".format(depth_ = depth, bf_ = bit_flips_per_lvl)
				))
				cf_correctionfactor = (1, 1, 1, 1, 1, 1, 1, 1, 1)

			rs_for_lbl_plots = ()

		elif args.result_battery=="influence_of_tree_depth_test":
			rs_names = {}
			accuracy_to_compare = []
			accuracy_plot_style = "comp"
			bit_flips_per_lvl = int(args.bf_ratio*args.artificial_seq_len)
			fs_name = "influence_of_tree_depth_mixed_{bf_:d}bits".format(
				bf_ = bit_flips_per_lvl
			)

			n_tree_depths = 3
			for depth_id, depth in enumerate((4,5)):
				rs_names["artificial_d{depth_:d}UltraMixed{bf_:d}bits".format(depth_ = depth, bf_ = bit_flips_per_lvl)] = depth_id/(n_tree_depths+0.33)
				rs_names["artificial_d{depth_:d}RbMixed{bf_:d}bits".format(depth_ = depth, bf_ = bit_flips_per_lvl)] = (depth_id+0.33)/(n_tree_depths+0.33)
				rs_names["artificial_d{depth_:d}Unif{bf_:d}bits".format(depth_ = depth, bf_ = bit_flips_per_lvl)] = 0
				
				accuracy_to_compare.append((
					"artificial_d{depth_:d}RbMixed{bf_:d}bits".format(depth_ = depth, bf_ = bit_flips_per_lvl),
					"artificial_d{depth_:d}UltraMixed{bf_:d}bits".format(depth_ = depth, bf_ = bit_flips_per_lvl),
					"artificial_d{depth_:d}Unif{bf_:d}bits".format(depth_ = depth, bf_ = bit_flips_per_lvl)
				))
				cf_correctionfactor = (1, 1, 1, 1, 1, 1)

			rs_for_lbl_plots = ()

		elif args.result_battery=="influence_of_tree_depth_unmixed":
			rs_names = {}
			accuracy_to_compare = []
			accuracy_plot_style = "comp"
			bit_flips_per_lvl = int(args.bf_ratio*args.artificial_seq_len)
			fs_name = "influence_of_tree_depth_unmixed_{bf_:d}bits".format(
				bf_ = bit_flips_per_lvl
			)

			n_tree_depths = 3
			for depth_id, depth in enumerate((4,5,6)):
				rs_names["artificial_d{depth_:d}UltraUnmixed{bf_:d}bits".format(depth_ = depth, bf_ = bit_flips_per_lvl)] = depth_id/(n_tree_depths+0.33)
				rs_names["artificial_d{depth_:d}RbUnmixed{bf_:d}bits".format(depth_ = depth, bf_ = bit_flips_per_lvl)] = (depth_id+0.33)/(n_tree_depths+0.33)
				rs_names["artificial_d{depth_:d}Unif{bf_:d}bits".format(depth_ = depth, bf_ = bit_flips_per_lvl)] = 0
				
				accuracy_to_compare.append((
					"artificial_d{depth_:d}RbUnmixed{bf_:d}bits".format(depth_ = depth, bf_ = bit_flips_per_lvl),
					"artificial_d{depth_:d}UltraUnmixed{bf_:d}bits".format(depth_ = depth, bf_ = bit_flips_per_lvl),
					"artificial_d{depth_:d}Unif{bf_:d}bits".format(depth_ = depth, bf_ = bit_flips_per_lvl)
				))
				cf_correctionfactor = (1, 1, 1, 1, 1, 1, 1.25, 1, 1)

			rs_for_lbl_plots = ()

		elif args.result_battery=="influence_of_tree_depth_mixed_fixedExtremaTimeConstants":
			rs_names = {}
			accuracy_to_compare = []
			accuracy_plot_style = "comp"
			bit_flips_per_lvl = int(args.bf_ratio*args.artificial_seq_len)
			fs_name = "influence_of_tree_depth_mixed_{bf_:d}bits".format(
				bf_ = bit_flips_per_lvl
			)

			n_tree_depths = 3
			for depth_id, depth in enumerate((4,5,6)):
				rs_names["artificial_d{depth_:d}UltraMixed{bf_:d}bits".format(depth_ = depth, bf_ = bit_flips_per_lvl)] = depth_id/(n_tree_depths+0.33)
				rs_names["artificial_d{depth_:d}RbMixed{bf_:d}bits".format(depth_ = depth, bf_ = bit_flips_per_lvl)] = (depth_id+0.33)/(n_tree_depths+0.33)
				rs_names["artificial_d{depth_:d}Unif{bf_:d}bits".format(depth_ = depth, bf_ = bit_flips_per_lvl)] = 0
				
				accuracy_to_compare.append((
					"artificial_d{depth_:d}RbMixed{bf_:d}bits".format(depth_ = depth, bf_ = bit_flips_per_lvl),
					"artificial_d{depth_:d}UltraMixed{bf_:d}bits".format(depth_ = depth, bf_ = bit_flips_per_lvl),
					"artificial_d{depth_:d}Unif{bf_:d}bits".format(depth_ = depth, bf_ = bit_flips_per_lvl)
				))

			rs_for_lbl_plots = ()
			cf_correctionfactor = None

		elif args.result_battery=="influence_of_tree_depth_unmixed_fixedExtremaTimeConstants":
			rs_names = {}
			accuracy_to_compare = []
			accuracy_plot_style = "comp"
			bit_flips_per_lvl = int(args.bf_ratio*args.artificial_seq_len)
			fs_name = "influence_of_tree_depth_unmixed_{bf_:d}bits".format(
				bf_ = bit_flips_per_lvl
			)

			n_tree_depths = 3
			for depth_id, depth in enumerate((4,5,6)):
				rs_names["artificial_d{depth_:d}UltraUnmixed{bf_:d}bits".format(depth_ = depth, bf_ = bit_flips_per_lvl)] = depth_id/(n_tree_depths+0.33)
				rs_names["artificial_d{depth_:d}RbUnmixed{bf_:d}bits".format(depth_ = depth, bf_ = bit_flips_per_lvl)] = (depth_id+0.33)/(n_tree_depths+0.33)
				rs_names["artificial_d{depth_:d}Unif{bf_:d}bits".format(depth_ = depth, bf_ = bit_flips_per_lvl)] = 0
				
				accuracy_to_compare.append((
					"artificial_d{depth_:d}RbUnmixed{bf_:d}bits".format(depth_ = depth, bf_ = bit_flips_per_lvl),
					"artificial_d{depth_:d}UltraUnmixed{bf_:d}bits".format(depth_ = depth, bf_ = bit_flips_per_lvl),
					"artificial_d{depth_:d}Unif{bf_:d}bits".format(depth_ = depth, bf_ = bit_flips_per_lvl)
				))

			rs_for_lbl_plots = ()
			cf_correctionfactor = None

	elif args.dataset == 'MNIST':
		blocks = MNIST_blocks
		cf_correctionfactor = None
		if args.result_battery=="ultra_vs_rb2_mixed":
			if args.optimizer=="sgd":
				fs_name = "mlp_sgd"
				rs_names = {
					"MNIST_UltraMixed": 0.6,
					"MNIST_RbMixed": 0.3,
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
			
			if args.optimizer == "adam":
				fs_name = "mlp_adam"
				rs_names = {
					"MNIST_UltraMixed": 0.6,
					"MNIST_RbMixed": 0.3,
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

			if args.optimizer == "adagrad":
				fs_name = "mlp_adam"
				rs_names = {
					"MNIST_UltraMixed": 0.6,
					"MNIST_RbMixed": 0.3,
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
	fs = FigureSet(fs_name=fs_name, rs_names=rs_names, accuracy_to_compare=accuracy_to_compare, accuracy_plot_style=accuracy_plot_style, rs_for_lbl_plots=rs_for_lbl_plots, artificial_seq_len=args.artificial_seq_len, cf_correctionfactor=cf_correctionfactor, draw_timescales=args.draw_timescales, draw_explorations=args.draw_explorations)
	print("Done")

	if args.make_lbl_history < 2:
		print("Making CF-related figures...")
		make_CFfigures(fs, blocks, save_formats=['svg', 'pdf'], acc_mode=args.acc_mode, acc_max_iter=args.acc_max_iter, cfprof_x_origpos=args.cfprof_x_origpos, cf_conf=args.cf_confidence, cfprof_ymax=args.cfprof_ymax, rb2_norm=args.rb2_norm, fit_um_profile=args.fit_um_profile, fit_rb2_profile=args.fit_rb2_profile)
		print("Done")

	if args.make_lbl_history:
		for rs in fs.rs_for_lbl_plots:
			print("Making label history figures...")
			rs.lbl_distrib(max_iter=args.first_iters_focus, cumulative=True, save_formats=['svg', 'pdf'], multi_simus=False)
			rs.lbl_distrib(max_iter=args.first_iters_focus, cumulative=False, save_formats=['svg', 'pdf'], multi_simus=False)
			print("Done")