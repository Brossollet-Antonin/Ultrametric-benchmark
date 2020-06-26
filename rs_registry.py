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
	'MNIST': {
		'tree_depth': 3,
		'dataset': 'MNIST_8',
		'nnarchi': 'FCL256',
		'T': 0.225,
		'shuffle_size': 1000,
		'seq_length': 1000000,
		'n_tests': 150
	},
	'artificial_d3': {
		'tree_depth': 3,
		'dataset': 'artificial_8',
		'nnarchi': 'FCL20',
		'T': 0.225,
		'shuffle_size': 1000,
		'seq_length': 300000,
		'n_tests': 300,
		'artificial_seq_len': 200,
		'bf_ratios': (0.1, 0.13)
	},
	'artificial_d4': {
		'tree_depth': 4,
		'dataset': 'artificial_16',
		'nnarchi': 'FCL10',
		'T': 0.4,
		'shuffle_size': 150,
		'seq_length': 300000,
		'n_tests': 300,
		'artificial_seq_len': 200,
		'bf_ratios': (0.1)
	},
	'artificial_d5': {
		'tree_depth': 5,
		'dataset': 'artificial_32',
		'nnarchi': 'FCL20',
		'T': 0.4,
		'shuffle_size': 1000,
		'seq_length': 300000,
		'n_tests': 300,
		'artificial_seq_len': 200,
		'bf_ratios': (0.04, 0.07, 0.1, 0.13)
	},
	'artificial_d6': {
		'tree_depth': 6,
		'dataset': 'artificial_64',
		'nnarchi': 'FCL40',
		'T': 0.4,
		'shuffle_size': 2500,
		'seq_length': 900000,
		'n_tests': 300,
		'artificial_seq_len': 200,
		'bf_ratios': (0.1)
	},
	'artificial_d7': {
		'tree_depth': 7,
		'dataset': 'artificial_128',
		'nnarchi': 'FCL50',
		'T': 0.6,
		'shuffle_size': 328,
		'seq_length': 300000,
		'n_tests': 300,
		'artificial_seq_len': 200,
		'bf_ratios': (0.04, 0.07, 0.1, 0.13)
	},
	'artificial_d10': {
		'tree_depth': 10,
		'dataset': 'artificial_1024',
		'nnarchi': 'FCL100',
		'T': 0.85,
		'shuffle_size': 328,
		'seq_length': 300000,
		'n_tests': 300,
		'artificial_seq_len': 200,
		'bf_ratios': (0.04,)
	}
}

name_to_descr = {
	'Ultra': 'Ultrametric',
	'Rb': 'Random blocks',
	'Unif': 'Uniform',
	'd3': 'depth 3',
	'd5': 'depth 5',
	'd7': 'depth 7',
	'd10': 'depth 10',
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

RS_DIR = {}

## Instanciating ResultSet objects for artificial datasets
for depth in ('d3', 'd4', 'd5', 'd6', 'd7', 'd10'):
	paramset = params["artificial_{:s}".format(depth)]
	for bf_ratio in paramset['bf_ratios']:
		bit_flips_per_lvl = int(bf_ratio*paramset["artificial_seq_len"])
		for seq_type in ('Ultra', 'Rb'):
			for leaves_mix in ('Mixed', 'Unmixed'):
				rs_name = "artificial_{depth_:s}{seq_type_:s}{leaves_mix_:s}{bit_flips_per_lvl_:d}bits".format(
					depth_ = depth,
					seq_type_ = seq_type,
					leaves_mix_ = leaves_mix,
					bit_flips_per_lvl_ = bit_flips_per_lvl
				)
				rs_descr = "Artificial - {seq_type_descr_:s} {depth_descr_:s} ({leaves_mix_descr_:s}, {bit_flips_per_lvl_:d}bits/lvl)".format(
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
					dataset_name = paramset["dataset"],
					nn_config = paramset["nnarchi"],
					seq_type = rs_path,
					seq_length = paramset["seq_length"],
					simset_id = paramset["T"] if seq_type=='Ultra' else paramset["shuffle_size"]
				)

		if depth == 'd10':
			continue

		unif_rs_name = "artificial_{depth_:s}Unif{bit_flips_per_lvl_:d}bits".format(
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
			dataset_name = paramset["dataset"],
			nn_config = paramset["nnarchi"],
			seq_type = unif_rs_path,
			seq_length = paramset["seq_length"],
			simset_id = 0.0
		)

## Instanciating ResultSet objects for MNIST
for seq_type in ('Ultra', 'Rb'):
	for leaves_mix in ['Mixed']:
		rs_name = "MNIST_{seq_type_:s}{leaves_mix_:s}".format(
			seq_type_ = seq_type,
			leaves_mix_ = leaves_mix
		)
		rs_descr = "MNIST - {seq_type_descr_:s} ({leaves_mix_descr_:s})".format(
			seq_type_descr_ = name_to_descr[seq_type],
			leaves_mix_descr_ = name_to_descr[leaves_mix]
		)
		rs_path = "{seq_type_path_:s}{leaves_mix_path_:s}".format(
			seq_type_path_ = name_to_path[seq_type],
			leaves_mix_path_ = name_to_path[leaves_mix]
		)

		RS_DIR[rs_name] = ld.ResultSet(
			rs_name = rs_name,
			rs_descr = rs_descr,
			sim_map_dict = sim_directory,
			dataset_name = params["MNIST"]["dataset"],
			nn_config = params["MNIST"]["nnarchi"],
			seq_type = rs_path,
			seq_length = paramset["seq_length"],
			simset_id = params["MNIST"]["T"] if seq_type=='Ultra' else params["MNIST"]["shuffle_size"]
		)

unif_rs_name = "MNIST_Unif"
unif_rs_descr = "MNIST - Uniform"
unif_rs_path = "uniform"

RS_DIR[unif_rs_name] = ld.ResultSet(
	rs_name = unif_rs_name,
	rs_descr = unif_rs_descr,
	sim_map_dict = sim_directory,
	dataset_name = params["MNIST"]["dataset"],
	nn_config = params["MNIST"]["nnarchi"],
	seq_type = unif_rs_path,
	seq_length = paramset["seq_length"],
	simset_id = 0.0
)

def get_directory_view():
	print("Current directory\n")
	for rs_name, rs in RS_DIR.items():
		print("{name_:s} | {descr_:s} @ {path_:s}".format(
			name_ = rs_name,
			descr_ = rs.descr,
			path_ = rs.seq_type
		))