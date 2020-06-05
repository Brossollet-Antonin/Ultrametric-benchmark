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
	'artificial_d5': {
		'tree_depth': 5,
		'dataset': 'artificial_32',
		'nnarchi': 'FCL20',
		'T': 0.4,
		'shuffle_size': 1000,
		'seq_length': 300000,
		'n_tests': 300,
		'artificial_seq_len': 200
	},
	'artificial_d7': {
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

RS_DIR = {}

## Instanciating ResultSet objects for artificial datasets
for depth in ('d5', 'd7'):
	for bf_ratio in (0.04, 0.07, 0.1, 0.13):
		bit_flips_per_lvl = int(bf_ratio*params["artificial_{:s}".format(depth)]["artificial_seq_len"])
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
					dataset_name = params["artificial_{:s}".format(depth)]["dataset"],
					nn_config = params["artificial_{:s}".format(depth)]["nnarchi"],
					seq_type = rs_path,
					simset_id = params["artificial_{:s}".format(depth)]["T"] if seq_type=='Ultra' else params["artificial_{:s}".format(depth)]["shuffle_size"]
				)

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
			dataset_name = params["artificial_{:s}".format(depth)]["dataset"],
			nn_config = params["artificial_{:s}".format(depth)]["nnarchi"],
			seq_type = unif_rs_path,
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
	simset_id = 0.0
)