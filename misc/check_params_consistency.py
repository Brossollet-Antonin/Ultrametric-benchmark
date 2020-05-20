import os, sys
sys.path.append('../')

import numpy as np
from scipy.io import savemat
import pdb
import utils
import ast, json
from distutils.dir_util import copy_tree
import argparse

paths = utils.get_project_paths()
um_root = paths['simus']+'1toM/artificial_32/FCL20/'

parser = argparse.ArgumentParser(os.path.join(paths['misc'], 'check_params_consistency.py'), description='Checking that each simulation set uses the same parameters, to ensure concsitency of the results')
parser.add_argument('--simuset_root_list', type=str, nargs='*', help="Simulation set folders that will be checked (internally, independently from one another)")

args = parser.parse_args()

for simuset in args.simuset_root_list:
    print("Checking parameters consistency for simuset {:s}".format(simuset))
    simuset_root = os.path.join(um_root, simuset)
    simus_root = os.listdir(simuset_root)
    bynflips_sorter = {}
    simu_ref_subdir = simus_root[0]
    with open(os.path.join(simuset_root, simu_ref_subdir, 'parameters.json'), 'r') as ref_param_file:
        ref_params = json.loads(ref_param_file.read())

    print("    Reference taken for parameters: {:s}".format(os.path.join(simuset_root, simu_ref_subdir)))
    print(json.dumps(ref_params, indent=2))

    params_that_should_match = [
        "Flips ratio",
        "Temperature",
        "Tree Depth",
        "Tree Branching",
        "Flips ratio",
        "Sequence Length",
        "Minibatches Size",
        "Number of tests",
        "Energy Step",
        "Replay Memory Size",
        "Learning rate",
        "Dataset",
        "device_type",
        "NN architecture",
        "Split total length",
        "Original command"
    ]

    for simu_subdir in simus_root:
        # For all nflips simulated, list the corresponding simulation subfolders
        with open(os.path.join(simuset_root, simu_subdir, 'parameters.json'), 'r') as param_file:
            simu_params = json.loads(param_file.read())

        for param_name in params_that_should_match:
            assert simu_params[param_name] == ref_params[param_name], "    {0:s} MISMATCH BETWEEN {1:s} AND {2:s}".format(param_name, simu_subdir, simu_ref_subdir)

    print("Simulation set {:s} is consistent! Producing a single parameters file for the simulation set...".format(simuset))
    simuset_params = {k: ref_params[k] for k in params_that_should_match}
    with open(os.path.join(simuset_root, "parameters.json"), 'w') as outfile:
        json.dump(simuset_params, outfile)
    print("done!\n\n")
