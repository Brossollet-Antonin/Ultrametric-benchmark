import os, sys
sys.path.append('../')

import numpy as np
import utils
import json
import argparse

paths = utils.get_project_paths()

parser = argparse.ArgumentParser(os.path.join(paths['misc'], 'check_params_consistency.py'), description='Checking that each simulation set uses the same parameters, to ensure concsitency of the results')
parser.add_argument('--simuset_root_dir', type=str, help="Root to the folder simulation set folders that will be checked (internally, independently from one another)")
parser.add_argument('--remove_eval_files', action='store_true', default=False, help="If specified, will delete the (usually heavy) evaluation files that were created by default at simulation time before 06/27/20")

args = parser.parse_args()

for simuset in os.listdir(args.simuset_root_dir):
    if not os.path.isdir(os.path.join(args.simuset_root_dir, simuset)):
        continue

    print("Checking parameters consistency for simuset {:s}".format(simuset))
    simuset_path = os.path.join(args.simuset_root_dir, simuset)
    simus_path = [simu_path for simu_path in os.listdir(simuset_path) if os.path.isdir(os.path.join(simuset_path, simu_path))]

    simu_ref_subdir = simus_path[0]
    with open(os.path.join(simuset_path, simu_ref_subdir, 'parameters.json'), 'r') as ref_param_file:
        ref_params = json.loads(ref_param_file.read())

    print("    Reference taken for parameters: {:s}".format(os.path.join(simuset_path, simu_ref_subdir)))
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
        "Original command",
        "Timescales"
    ]

    for simu_subdir in simus_path:
        # For all nflips simulated, list the corresponding simulation subfolders
        with open(os.path.join(simuset_path, simu_subdir, 'parameters.json'), 'r') as param_file:
            simu_params = json.loads(param_file.read())

        for param_name in params_that_should_match:
            if (param_name in simu_params.keys()) and (param_name in ref_params.keys()):
                assert simu_params[param_name] == ref_params[param_name], "    {0:s} MISMATCH BETWEEN {1:s} AND {2:s}".format(param_name, simu_subdir, simu_ref_subdir)

        if args.remove_eval_files:
            # Original file
            if os.path.exists(os.path.join(simuset_path, simu_subdir, 'evaluation_original.npy')):
                os.remove(os.path.join(simuset_path, simu_subdir, 'evaluation_original.npy'))

            # Shuffle files
            shuffle_dirs = [shuffle_dir for shuffle_dir in os.listdir(os.path.join(simuset_path, simu_subdir)) if os.path.isdir(os.path.join(simuset_path, simu_subdir, shuffle_dir))]
            for shuffle_dir in shuffle_dirs:
                if 'shuffle_' not in shuffle_dir:
                    continue
                if os.path.exists(os.path.join(simuset_path, simu_subdir, shuffle_dir, 'evaluation_shuffled.npy')):
                    os.remove(os.path.join(simuset_path, simu_subdir, shuffle_dir, 'evaluation_shuffled.npy'))
            print("Removed evaluation files from simulation result folders")

    print("Simulation set {:s} is consistent! Producing a single parameters file for the simulation set...".format(simuset))
    simuset_params = {k: ref_params[k] for k in params_that_should_match if k in ref_params.keys()}
    with open(os.path.join(simuset_path, "parameters.json"), 'w') as outfile:
        json.dump(simuset_params, outfile)
    print("done!\n\n")
