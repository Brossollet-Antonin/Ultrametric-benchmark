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

# ToDo: Refactoring - pour chaque dataset, stocker des paramètres linear et des paramètres celu séparémment
params = { 
	'MNIST': {
		'tree_depth': 3,
		'dataset': 'MNIST_8',
		'T': 0.225,
		'shuffle_size': 360,
		'linear_config': {
			'nnarchi': 'FCL256_celu',
			'seq_length': 4000000,
			'n_tests': 500
		},
		'celu_config': {
			'nnarchi': 'FCL256_linear',
			'seq_length': 4000000,
			'n_tests': 500
		}
	},
	'artificial_d3': {
		'tree_depth': 3,
		'dataset': 'artificial_8',
		'T': 0.225,
		'shuffle_size': 1000,
		'artificial_seq_len': 200,
		'bf_ratios': (0.1, 0.13),
		'linear_config': {
			'nnarchi': 'FCL20',
			'seq_length': 300000,
			'n_tests': 300
		},
		'celu_config': {}
	},
	'artificial_d4': {
		'tree_depth': 4,
		'dataset': 'artificial_16',
		'T': 0.4,
		'shuffle_size': 150,
		'artificial_seq_len': 200,
		'bf_ratios': (0.1,),
		'celu_config': {
			'nnarchi': 'FCL10_celu',
			'seq_length': 240000, #60000,
			'n_tests': 480 #240
		},
		'linear_config': {
			'nnarchi': 'FCL10_linear',
			'seq_length': 30000,
			'n_tests': 300
		}
	},
	'artificial_d5': {
		'tree_depth': 5,
		'dataset': 'artificial_32',
		'T': 0.4,
		'shuffle_size': 1000,
		'artificial_seq_len': 200,
		'bf_ratios': (0.04, 0.07, 0.1, 0.13),
		'celu_config': {
			'nnarchi': 'FCL20_celu',
			'seq_length': 600000,
			'n_tests': 600
		},
		'linear_config': {
			'nnarchi': 'FCL20_linear',
			'seq_length': 300000,
			'n_tests': 300
		}
	},
	'artificial_d6': {
		'tree_depth': 6,
		'dataset': 'artificial_64',
		'T': 0.4,
		'shuffle_size': 5000,
		'artificial_seq_len': 200,
		'bf_ratios': (0.1,),
		'celu_config': {
			'nnarchi': 'FCL40_celu',
			'seq_length': 4000000,
			'n_tests': 800
		},
		'linear_config': {
			'nnarchi': 'FCL40_linear',
			'seq_length': 900000,
			'n_tests': 300
		}
	},
	'artificial_d7': {
		'tree_depth': 7,
		'dataset': 'artificial_128',
		'T': 0.6,
		'shuffle_size': 328,
		'artificial_seq_len': 200,
		'bf_ratios': (0.04, 0.07, 0.1, 0.13),
		'linear_config': {
			'nnarchi': 'FCL50',
			'seq_length': 300000,
			'n_tests': 300
		},
		'celu_config': {}
		
	},
	'artificial_d10': {
		'tree_depth': 10,
		'dataset': 'artificial_1024',
		'T': 0.85,
		'shuffle_size': 328,
		'artificial_seq_len': 200,
		'bf_ratios': (0.04,),
		'linear_config': {
			'nnarchi': 'FCL100',
			'seq_length': 300000,
			'n_tests': 300
		},
		'celu_config': {}
	}
}

name_to_descr = {
	'Ultra': 'Ultrametric',
	'Rb': 'Random blocks',
	'Unif': 'Uniform',
	'd3': 'depth 3',
	'd4': 'depth 4',
	'd5': 'depth 5',
	'd6': 'depth 6',
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
for depth in ('d4', 'd5', 'd6'):
	paramset = params["artificial_{:s}".format(depth)]
	for bf_ratio in paramset['bf_ratios']:
		bit_flips_per_lvl = int(bf_ratio*paramset["artificial_seq_len"])
		for leaves_mix in ('Mixed', 'Unmixed'):
			for nonlin in ["linear", "celu"]:
				nonlin_config = "{nonlin_:s}_config".format(nonlin_ = nonlin)
				for seq_type in ('Ultra', 'Rb'):
					rs_name = "artificial_{nonlin_:s}_{depth_:s}{seq_type_:s}{leaves_mix_:s}{bit_flips_per_lvl_:d}bits".format(
						nonlin_ = nonlin,
						depth_ = depth,
						seq_type_ = seq_type,
						leaves_mix_ = leaves_mix,
						bit_flips_per_lvl_ = bit_flips_per_lvl
					)
					rs_descr = "Artificial - {seq_type_descr_:s} {depth_descr_:s}".format(
						seq_type_descr_ = name_to_descr[seq_type],
						depth_descr_ = name_to_descr[depth]
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
						nn_config = paramset[nonlin_config]["nnarchi"],
						seq_type = rs_path,
						seq_type_descr = name_to_descr[seq_type],
						seq_length = paramset[nonlin_config]["seq_length"],
						simset_id = paramset["T"] if seq_type=='Ultra' else paramset["shuffle_size"]
					)

				if depth == 'd10':
					continue

				unif_rs_name = "artificial_{nonlin_:s}_{depth_:s}Unif{bit_flips_per_lvl_:d}bits".format(
					nonlin_ = nonlin,
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
					nn_config = paramset[nonlin_config]["nnarchi"],
					seq_type = unif_rs_path,
					seq_type_descr = "Uniform",
					seq_length = paramset[nonlin_config]["seq_length"],
					simset_id = 0.0
				)

## Instanciating ResultSet objects for MNIST
for seq_type in ('Ultra', 'Rb'):
	for nonlin in ["linear", "celu"]:
		nonlin_config = "{nonlin_:s}_config".format(nonlin_ = nonlin)
		rs_name = "MNIST_{seq_type_:s}".format(seq_type_ = seq_type)
		rs_descr = "MNIST - {seq_type_descr_:s}".format(seq_type_descr_ = name_to_descr[seq_type])
		rs_path = "{seq_type_path_:s}".format(seq_type_path_ = name_to_path[seq_type])

		RS_DIR[rs_name] = ld.ResultSet(
			rs_name = rs_name,
			rs_descr = rs_descr,
			sim_map_dict = sim_directory,
			dataset_name = params["MNIST"]["dataset"],
			nn_config = params["MNIST"][nonlin_config]["nnarchi"],
			seq_type = rs_path,
			seq_type_descr = name_to_descr[seq_type],
			seq_length = params["MNIST"][nonlin_config]["seq_length"],
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
	nn_config = params["MNIST"][nonlin_config]["nnarchi"],
	seq_type = unif_rs_path,
	seq_type_descr = "Uniform",
	seq_length = params["MNIST"][nonlin_config]["seq_length"],
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