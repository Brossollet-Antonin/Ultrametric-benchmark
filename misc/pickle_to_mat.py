	import os, sys
sys.path.append('../')

import numpy as np
from scipy.io import savemat
import utils
import argparse

paths = utils.get_project_paths()

parser = argparse.ArgumentParser(os.path.join(paths['misc'], 'pickle_to_mat.py'), description='Transcrubes all pickle simulation outputs to mat files for autocorrelation to be computed (much faster in Matlab)')
parser.add_argument('--simusets_root', type=str, help="Root to the folder simulation set folders that will be checked (internally, independently from one another). Ex: '1toM/artificial_32/FCL20/'")

args = parser.parse_args()

simusets_types_folders = [simuset_type_folder for simuset_type_folder in os.listdir(os.path.join(paths['simus'], args.simusets_root)) if os.path.isdir(os.path.join(paths['simus'], args.simusets_root, simuset_type_folder))]

for simuset_type_folder in simusets_types_folders:
	print("Generating .mat files for {:s}".format(simuset_type_folder))
	simuset_type_folder = os.path.join(paths['simus'], args.simusets_root, simuset_type_folder)
	simuset_paths = [os.path.join(simuset_type_folder, simuset_path) for simuset_path in os.listdir(simuset_type_folder) if os.path.isdir(os.path.join(simuset_type_folder, simuset_path))]
	for simuset_path in simuset_paths:
		orig_seq_pickle = np.load(
			os.path.join(simuset_path, 'train_labels_orig.pickle'), allow_pickle=True
		)
		if not os.path.exists(os.path.join(simuset_path, 'matlab')):
			os.makedirs(os.path.join(simuset_path, 'matlab'))
		savemat(
			os.path.join(simuset_path, 'matlab', 'train_labels_orig.mat'),
			{ 'sequence': orig_seq_pickle }
		)

		shuffle_dirs = [shuffle_dir for shuffle_dir in os.listdir(simuset_path) if os.path.isdir(os.path.join(simuset_path, shuffle_dir))]
		for shuffle_dir in shuffle_dirs:
			if 'shuffle_' not in shuffle_dir:
				continue
			shfl_size = int(shuffle_dir.split('_')[1])
			shfl_seq_pickle = np.load(
				os.path.join(simuset_path, shuffle_dir, 'train_labels_shfl.pickle'),
				allow_pickle=True
			)
			savemat(
				os.path.join(simuset_path, 'matlab', 'train_labels_shfl_{:d}.mat'.format(shfl_size)),
				{ 'sequence': shfl_seq_pickle }
			)