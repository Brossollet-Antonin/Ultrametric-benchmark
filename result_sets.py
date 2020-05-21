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
sim_directory = utils.get_simus_directory()

import result_loader as ld

params = {
	'd5': {
		'tree_depth': 5,
		'dataset': 'artificial_32',
		'nnarchi': 'FCL20',
		'T': 0.4,
		'shuffle_size': 1000,
		'seq_length': 300000,
		'n_tests': 300,
		'artificial_seq_len': 200
	},
	'd7': {
		'tree_depth': 7,
		'dataset': 'artificial_128',
		'nnarchi': 'FCL50',
		'T': 0.6,
		'shuffle_size': 328,
		'seq_length': 300000,
		'n_tests': 300,
		'artificial_seq_len': 200
	}
}

name_to_descr = {
	'Ultra': 'Ultrametric',
	'Rb': 'Random blocks',
	'Unif': 'Uniform',
	'd5': 'depth 5',
	'd7': 'depth 7',
	'Mixed': 'mixed tree leaves',
	'Unmixed': 'unmixed tree leaves'
}
name_to_path = {
	'Ultra': 'ultrametric',
	'Rb': 'random_blocks2',
	'Unif': 'uniform',
	'Mixed': '',
	'Unmixed': '_noclshfl'
}

## Instanciating ResultSet objects of depth 5 (without loading any actual result data)
RS_DIR = {}

for depth in ('d5', 'd7'):
	for bf_ratio in (0.04, 0.07, 0.1, 0.13):
		bit_flips_per_lvl = int(bf_ratio*params[depth]["artificial_seq_len"])
		for seq_type in ('Ultra', 'Rb'):
			for leaves_mix in ('Mixed', 'Unmixed'):
				rs_name = "{depth_:s}{seq_type_:s}{leaves_mix_:s}{bit_flips_per_lvl_:d}bits".format(
					depth_ = depth,
					seq_type_ = seq_type,
					leaves_mix_ = leaves_mix,
					bit_flips_per_lvl_ = bit_flips_per_lvl
				)
				rs_descr = "{seq_type_descr_:s} {depth_descr_:s} ({leaves_mix_descr_:s}, {bit_flips_per_lvl_:d}bits/lvl)".format(
					seq_type_descr_ = name_to_descr[seq_type],
					depth_descr_ = name_to_descr[depth],
					leaves_mix_descr_ = name_to_descr[leaves_mix],
					bit_flips_per_lvl_ = bit_flips_per_lvl	
				)
				rs_path = "{seq_type_path_:s}_ratio{bit_flips_per_lvl_:d}{leaves_mix_path_:s}".format(
					seq_type_path_ = name_to_path[seq_type],
					leaves_mix_path_ = name_to_path[leaves_mix],
					bit_flips_per_lvl_ = bit_flips_per_lvl	
				)

				RS_DIR[rs_name] = ld.ResultSet(
					rs_name = rs_name,
					rs_descr = rs_descr,
					sim_map_dict = sim_directory,
					dataset_name = params[depth]["dataset"],
					nn_config = params[depth]["nnarchi"],
					seq_type = rs_path,
					simset_id = params[depth]["T"] if seq_type=='Ultra' else params[depth]["shuffle_size"]
				)

		unif_rs_name = "{depth_:s}Unif{bit_flips_per_lvl_:d}bits".format(
			depth_ = depth,
			bit_flips_per_lvl_ = bit_flips_per_lvl
		)
		unif_rs_descr = "Uniform {depth_descr_:s} ({bit_flips_per_lvl_:d}bits/lvl)".format(
			depth_descr_ = name_to_descr[depth],
			bit_flips_per_lvl_ = bit_flips_per_lvl
		)
		unif_rs_path = "uniform_ratio{bit_flips_per_lvl_:d}".format(
			bit_flips_per_lvl_ = bit_flips_per_lvl
		)

		RS_DIR[unif_rs_name] = ld.ResultSet(
			rs_name = unif_rs_name,
			rs_descr = unif_rs_descr,
			sim_map_dict = sim_directory,
			dataset_name = params[depth]["dataset"],
			nn_config = params[depth]["nnarchi"],
			seq_type = unif_rs_path,
			simset_id = 0.0
		)